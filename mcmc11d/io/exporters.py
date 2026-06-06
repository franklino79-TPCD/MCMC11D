"""
Exportadores de resultados a CSV y formatos tabulares.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd

from mcmc11d.physics.rg_flow import ResultadoRGFlow


def exportar_search(
    estados: jnp.ndarray,
    errores: jnp.ndarray,
    n_max: int,
    epsilon: float,
    batch_size: int,
    n_total: int,
    k_total: int,
    throughput: float,
    dt_total: float,
    logs_dir: Path,
) -> tuple[Path, Path]:
    """Exporta candidatos y metadatos del modo search."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    estados_np = jax.device_get(estados)
    errores_np = jax.device_get(errores)

    df = pd.DataFrame({
        "n1": estados_np[:, 0].astype(int) if k_total > 0 else [],
        "n2": estados_np[:, 1].astype(int) if k_total > 0 else [],
        "n3": estados_np[:, 2].astype(int) if k_total > 0 else [],
        "mse": errores_np if k_total > 0 else [],
    })
    if k_total > 0:
        df = df.sort_values("mse").reset_index(drop=True)

    path_cand = logs_dir / f"search_candidatos_{ts}.csv"
    df.to_csv(path_cand, index=False)

    tasa = k_total / n_total if n_total > 0 else 0.0
    df_meta = pd.DataFrame([{
        "timestamp": ts,
        "n_max": n_max,
        "epsilon": epsilon,
        "batch_size": batch_size,
        "gauge_fixed": True,
        "n1_fixed": 1,
        "espacio_efectivo": f"{{1}}x[1,{n_max}]^2",
        "n_estados": n_total,
        "n_candidatos": k_total,
        "tasa_supervivencia": tasa,
        "throughput_M_per_s": round(throughput, 4),
        "tiempo_total_s": round(dt_total, 3),
    }])
    path_meta = logs_dir / f"search_meta_{ts}.csv"
    df_meta.to_csv(path_meta, index=False)

    return path_cand, path_meta


def exportar_rg_flow(
    resultado: ResultadoRGFlow,
    n_max: int,
    mu_start: float,
    mu_end: float,
    mu_steps: int,
    alpha: float,
    logs_dir: Path,
) -> tuple[Path, Path]:
    """Exporta trayectoria RG y metadatos."""
    import dataclasses

    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    records = [dataclasses.asdict(r) for r in resultado.registros]
    df_traj = pd.DataFrame(records)
    path_traj = logs_dir / f"rg_flow_trayectoria_{ts}.csv"
    df_traj.to_csv(path_traj, index=False)

    df_meta = pd.DataFrame([{
        "timestamp": ts,
        "n_max": n_max,
        "mu_start": mu_start,
        "mu_end": mu_end,
        "mu_steps": mu_steps,
        "alpha": alpha,
        "n2_ir_final": resultado.n2_ir_final,
        "n3_ir_final": resultado.n3_ir_final,
        "beta_n2_mean_ir": resultado.beta_n2_mean_ir,
        "beta_n3_mean_ir": resultado.beta_n3_mean_ir,
        "punto_fijo_207_17": resultado.punto_fijo_207_17,
        "atractor_confirmado": resultado.atractor_confirmado,
        "parada_por_bekenstein": resultado.parada_por_bekenstein,
        "veredicto": resultado.veredicto,
        "tiempo_total_s": round(resultado.tiempo_total_s, 3),
    }])
    path_meta = logs_dir / f"rg_flow_meta_{ts}.csv"
    df_meta.to_csv(path_meta, index=False)

    return path_traj, path_meta
