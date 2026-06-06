"""
Orquestador del modo SEARCH — barrido sistematico gauge-fijo.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from mcmc11d.core.batching import (
    MicroBatch,
    estimar_vram_mb,
    generar_microbatches,
)
from mcmc11d.core.checkpoint import CheckpointManager, CheckpointState
from mcmc11d.core.constants import N1_GAUGE_FIXED
from mcmc11d.core.mesh import construir_malla_gauge_fija
from mcmc11d.io.exporters import exportar_search
from mcmc11d.physics.cost_engine import ResultadoFiltrado, filtrar_candidatos

log = logging.getLogger("mcmc11d")


def ejecutar_search(
    n_max: int,
    batch_size: int,
    epsilon: float,
    logs_dir: Path,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> None:
    """Barrido sistematico del espacio gauge-fijo {1} x [1, n_max]^2."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    n_total = n_max ** 2
    n_batches = math.ceil(n_total / batch_size)
    mem_malla = estimar_vram_mb(n_total)
    mem_batch = estimar_vram_mb(min(batch_size, n_total))

    log.info("Modo SEARCH iniciado  [GAUGE FIXING ACTIVO]")
    log.info("  n1 fijado (gauge)   : %d", N1_GAUGE_FIXED)
    log.info("  Espacio gauge-fijo  : {%d} x [1, %d]^2 = %s estados",
             N1_GAUGE_FIXED, n_max, f"{n_total:,}")
    log.info("  Reduccion vs [1,%d]^3: %.0fx", n_max, n_max)
    log.info("  Batches             : %d x %s estados", n_batches, f"{batch_size:,}")
    log.info("  Memoria malla       : %.1f MB  |  batch: %.1f MB", mem_malla, mem_batch)
    log.info("  Epsilon             : %.0e", epsilon)

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_mgr: CheckpointManager | None = None
    start_batch = 0
    candidatos_acum: list[jnp.ndarray] = []
    errores_acum: list[jnp.ndarray] = []

    if checkpoint_dir is not None:
        ckpt_mgr = CheckpointManager(checkpoint_dir)
        if not ckpt_mgr.adquirir_lock():
            log.error("No se pudo adquirir lock de checkpoint. Abortando.")
            return

        if resume:
            ckpt = ckpt_mgr.cargar_ultimo()
            if ckpt and ckpt.mode == "search" and ckpt.n_max == n_max:
                start_batch = ckpt.ultimo_batch + 1
                if ckpt.candidatos_n:
                    import numpy as np
                    candidatos_acum.append(
                        jnp.array(ckpt.candidatos_n, dtype=jnp.float32)
                    )
                    errores_acum.append(
                        jnp.array(ckpt.candidatos_mse, dtype=jnp.float32)
                    )
                log.info("Reanudando desde batch %d/%d", start_batch, n_batches)

    # ── Construir malla ───────────────────────────────────────────────────
    log.info("Construyendo malla gauge-fija...")
    t_malla = time.perf_counter()
    malla = construir_malla_gauge_fija(n_max)
    malla.block_until_ready()
    log.info("  Malla lista: shape=%s (%.0f ms)",
             malla.shape, 1000 * (time.perf_counter() - t_malla))

    # ── Barrido ───────────────────────────────────────────────────────────
    t_sweep = time.perf_counter()
    procesados = 0

    print()
    for mb in generar_microbatches(malla, batch_size):
        if mb.index < start_batch:
            continue

        t_b = time.perf_counter()
        resultado: ResultadoFiltrado = filtrar_candidatos(mb.data, epsilon=epsilon)
        dt_b = time.perf_counter() - t_b

        procesados += resultado.n_evaluados

        log.debug("  batch %3d/%d  estados=%s  hits=%d  throughput=%.2f M/s",
                   mb.index + 1, mb.total,
                   f"{resultado.n_evaluados:,}",
                   resultado.n_candidatos,
                   resultado.n_evaluados / dt_b / 1e6)

        if resultado.n_candidatos > 0:
            candidatos_acum.append(resultado.estados)
            errores_acum.append(resultado.errores)

        # Checkpoint periodico
        if ckpt_mgr and (mb.index + 1) % max(1, mb.total // 10) == 0:
            all_cands = jnp.concatenate(candidatos_acum) if candidatos_acum else jnp.empty((0, 3))
            all_errs = jnp.concatenate(errores_acum) if errores_acum else jnp.empty((0,))
            ckpt_mgr.guardar(CheckpointState(
                mode="search",
                ultimo_batch=mb.index,
                total_batches=mb.total,
                n_max=n_max,
                epsilon=epsilon,
                batch_size=batch_size,
                candidatos_n=jax.device_get(all_cands).astype(int).tolist() if all_cands.shape[0] > 0 else [],
                candidatos_mse=jax.device_get(all_errs).tolist() if all_errs.shape[0] > 0 else [],
            ))

        if (mb.index + 1) % max(1, mb.total // 10) == 0 or mb.index == mb.total - 1:
            pct = 100 * (mb.index + 1) / mb.total
            elapsed = time.perf_counter() - t_sweep
            tp = procesados / elapsed / 1e6
            hits_acc = sum(int(e.shape[0]) for e in errores_acum)
            print(
                f"  [{pct:5.1f}%]  batch {mb.index+1:>4}/{mb.total}"
                f"  procesados={procesados:>12,}"
                f"  candidatos_acum={hits_acc:>6,}"
                f"  throughput={tp:.2f} M/s",
                flush=True,
            )

    dt_total = time.perf_counter() - t_sweep
    tp_global = n_total / dt_total / 1e6

    # ── Consolidar ────────────────────────────────────────────────────────
    if candidatos_acum:
        estados_finales = jnp.concatenate(candidatos_acum, axis=0)
        errores_finales = jnp.concatenate(errores_acum, axis=0)
        k_total = int(estados_finales.shape[0])
    else:
        estados_finales = jnp.empty((0, 3), dtype=jnp.float32)
        errores_finales = jnp.empty((0,), dtype=jnp.float32)
        k_total = 0

    tasa = k_total / n_total

    print()
    print("=" * 65)
    print("  RESULTADOS — MODO SEARCH")
    print("=" * 65)
    print(f"  Estados evaluados   : {n_total:,}")
    print(f"  Candidatos totales  : {k_total:,}")
    print(f"  Tasa supervivencia  : {tasa:.4e}")
    print(f"  Tiempo total        : {dt_total:.2f} s")
    print(f"  Throughput global   : {tp_global:.2f} M estados/s")
    print("=" * 65)

    path_cand, path_meta = exportar_search(
        estados=estados_finales,
        errores=errores_finales,
        n_max=n_max,
        epsilon=epsilon,
        batch_size=batch_size,
        n_total=n_total,
        k_total=k_total,
        throughput=tp_global,
        dt_total=dt_total,
        logs_dir=logs_dir,
    )

    print(f"\n  Candidatos -> {path_cand.name}")
    print(f"  Metadatos  -> {path_meta.name}")
    print(f"  Directorio -> {logs_dir}\n")

    if 0 < k_total <= 20:
        import pandas as pd
        df = pd.read_csv(path_cand)
        print("  Top candidatos por MSE:")
        print(df.head(min(10, k_total)).to_string(index=False))
        print()

    if ckpt_mgr:
        ckpt_mgr.limpiar()

    log.info("Modo SEARCH completado.")
