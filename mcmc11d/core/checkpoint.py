"""
Sistema de checkpointing tolerante a fallos para Windows 11.

Estrategia de persistencia:
──────────────────────────
1. Estado serializado como JSON + arrays numpy comprimidos (.npz).
2. Escritura atomica: se escribe a archivo temporal, luego se renombra
   (os.replace es atomico en NTFS para el mismo volumen).
3. Doble checkpoint: se mantienen los ultimos 2 checkpoints validos
   para proteger contra corrupcion durante escritura.
4. Lock file con PID para evitar colisiones entre procesos.

Contrato de reanudacion:
  - El checkpoint almacena: ultimo_batch_completado, candidatos_acumulados,
    errores_acumulados, parametros de ejecucion, timestamp.
  - Al reanudar, se salta directamente al batch siguiente al ultimo
    completado, sin recalcular batches previos.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("mcmc11d")

_LOCK_SUFFIX = ".lock"
_CKPT_PREFIX = "mcmc11d_ckpt"


@dataclass
class CheckpointState:
    """Estado serializable de una ejecucion interrumpible."""
    mode: str
    ultimo_batch: int
    total_batches: int
    n_max: int
    epsilon: float
    batch_size: int
    candidatos_n: list[list[int]] = field(default_factory=list)
    candidatos_mse: list[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    parametros_extra: dict[str, Any] = field(default_factory=dict)

    @property
    def progreso(self) -> float:
        if self.total_batches == 0:
            return 1.0
        return (self.ultimo_batch + 1) / self.total_batches


class CheckpointManager:
    """Gestor de checkpoints con escritura atomica y rotacion."""

    def __init__(self, directory: Path, max_kept: int = 2):
        self._dir = directory
        self._max_kept = max_kept
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._dir / f"{_CKPT_PREFIX}{_LOCK_SUFFIX}"

    def _ckpt_path(self, slot: int) -> Path:
        return self._dir / f"{_CKPT_PREFIX}_{slot}.json"

    def _arrays_path(self, slot: int) -> Path:
        return self._dir / f"{_CKPT_PREFIX}_{slot}_arrays.npz"

    def adquirir_lock(self) -> bool:
        """Intenta adquirir el lock file. Retorna False si otro proceso lo tiene."""
        if self._lock_path.exists():
            try:
                with open(self._lock_path) as f:
                    data = json.load(f)
                pid = data.get("pid", -1)
                try:
                    os.kill(pid, 0)
                    log.warning("Lock activo por PID %d", pid)
                    return False
                except (OSError, ProcessLookupError):
                    log.info("Lock huerfano de PID %d, reclamando", pid)
            except (json.JSONDecodeError, KeyError):
                pass

        with open(self._lock_path, "w") as f:
            json.dump({"pid": os.getpid(), "timestamp": time.time()}, f)
        return True

    def liberar_lock(self) -> None:
        if self._lock_path.exists():
            self._lock_path.unlink(missing_ok=True)

    def guardar(self, state: CheckpointState) -> Path:
        """Escritura atomica con rotacion de slots."""
        slot = int(time.time() * 1000) % self._max_kept
        path_json = self._ckpt_path(slot)
        path_arrays = self._arrays_path(slot)

        state_dict = asdict(state)
        candidatos = state_dict.pop("candidatos_n")
        mse_vals = state_dict.pop("candidatos_mse")

        fd, tmp_json = tempfile.mkstemp(
            dir=str(self._dir), suffix=".tmp", prefix="ckpt_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state_dict, f, indent=2)
            os.replace(tmp_json, str(path_json))
        except BaseException:
            if os.path.exists(tmp_json):
                os.unlink(tmp_json)
            raise

        if candidatos:
            np.savez_compressed(
                str(path_arrays),
                candidatos=np.array(candidatos, dtype=np.int32),
                mse=np.array(mse_vals, dtype=np.float32),
            )

        log.debug("Checkpoint guardado: slot=%d batch=%d/%d",
                   slot, state.ultimo_batch, state.total_batches)
        return path_json

    def cargar_ultimo(self) -> CheckpointState | None:
        """Carga el checkpoint mas reciente valido."""
        mejor: CheckpointState | None = None
        mejor_ts = 0.0

        for slot in range(self._max_kept):
            path_json = self._ckpt_path(slot)
            if not path_json.exists():
                continue
            try:
                with open(path_json) as f:
                    data = json.load(f)
                ts = data.get("timestamp", 0.0)
                if ts <= mejor_ts:
                    continue

                candidatos: list[list[int]] = []
                mse_vals: list[float] = []
                path_arrays = self._arrays_path(slot)
                if path_arrays.exists():
                    arrs = np.load(str(path_arrays))
                    candidatos = arrs["candidatos"].tolist()
                    mse_vals = arrs["mse"].tolist()

                mejor = CheckpointState(
                    mode=data["mode"],
                    ultimo_batch=data["ultimo_batch"],
                    total_batches=data["total_batches"],
                    n_max=data["n_max"],
                    epsilon=data["epsilon"],
                    batch_size=data["batch_size"],
                    candidatos_n=candidatos,
                    candidatos_mse=mse_vals,
                    timestamp=ts,
                    parametros_extra=data.get("parametros_extra", {}),
                )
                mejor_ts = ts
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                log.warning("Checkpoint slot=%d corrupto: %s", slot, e)

        if mejor:
            log.info(
                "Checkpoint recuperado: batch %d/%d (%.1f%%)",
                mejor.ultimo_batch + 1,
                mejor.total_batches,
                mejor.progreso * 100,
            )
        return mejor

    def limpiar(self) -> None:
        """Elimina todos los checkpoints y el lock."""
        for slot in range(self._max_kept):
            for p in (self._ckpt_path(slot), self._arrays_path(slot)):
                p.unlink(missing_ok=True)
        self.liberar_lock()
