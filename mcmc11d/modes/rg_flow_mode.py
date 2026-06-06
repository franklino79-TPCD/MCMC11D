"""
Orquestador del modo RG-FLOW — Fase 3A con parada holografica.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcmc11d.io.exporters import exportar_rg_flow
from mcmc11d.physics.rg_flow import ejecutar_rg_flow

log = logging.getLogger("mcmc11d")


def ejecutar_rg_flow_mode(
    n_max: int,
    mu_start: float,
    mu_end: float,
    mu_steps: int,
    alpha: float,
    logs_dir: Path,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)

    resultado = ejecutar_rg_flow(
        n_max=n_max,
        mu_start=mu_start,
        mu_end=mu_end,
        mu_steps=mu_steps,
        alpha=alpha,
    )

    path_traj, path_meta = exportar_rg_flow(
        resultado=resultado,
        n_max=n_max,
        mu_start=mu_start,
        mu_end=mu_end,
        mu_steps=mu_steps,
        alpha=alpha,
        logs_dir=logs_dir,
    )

    print(f"\n  Trayectoria -> {path_traj.name}  ({len(resultado.registros)} filas)")
    print(f"  Metadatos   -> {path_meta.name}")
    print(f"  Directorio  -> {logs_dir}\n")

    log.info("Modo RG-FLOW completado. Atractor: %s", resultado.atractor_confirmado)
