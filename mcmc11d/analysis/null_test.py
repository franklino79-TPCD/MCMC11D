"""
Validacion estadistica Monte Carlo (null test de Fano).

Cuantifica la probabilidad de obtener candidatos por azar bajo
distribucion uniforme en [1, n_max]^3.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from mcmc11d.physics.cost_engine import ResultadoFiltrado, calcular_mse_batch

log = logging.getLogger("mcmc11d")


def ejecutar_null_test(
    n_samples: int,
    n_min: int,
    n_max: int,
    epsilon: float,
    seed: int,
) -> tuple[ResultadoFiltrado, float]:
    """Genera tripletes uniformes y evalua cuantos pasan el umbral epsilon.

    Returns
    -------
    (resultado, p_value) donde p_value = n_hits / n_samples
    """
    key = jax.random.PRNGKey(seed)
    tripletes = jax.random.randint(
        key, shape=(n_samples, 3),
        minval=n_min, maxval=n_max + 1,
    ).astype(jnp.float32)

    mse = calcular_mse_batch(tripletes)
    mse.block_until_ready()

    mascara = jax.device_get(mse < epsilon)
    hits = tripletes[mascara]
    mse_hits = mse[mascara]

    n_hits = int(hits.shape[0])
    p_value = n_hits / n_samples if n_samples > 0 else 0.0

    resultado = ResultadoFiltrado(
        estados=hits,
        errores=mse_hits,
        n_evaluados=n_samples,
        n_candidatos=n_hits,
    )
    return resultado, p_value


def exportar_resultados(
    resultado: ResultadoFiltrado,
    p_value: float,
    n_min: int,
    n_max: int,
    epsilon: float,
    seed: int,
    logs_dir: Path,
) -> tuple[Path, Path | None]:
    """Exporta resultados del null test a CSV."""
    import datetime
    import pandas as pd

    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df_summary = pd.DataFrame([{
        "timestamp": ts,
        "n_samples": resultado.n_evaluados,
        "n_min": n_min,
        "n_max": n_max,
        "epsilon": epsilon,
        "seed": seed,
        "n_hits": resultado.n_candidatos,
        "p_value": p_value,
    }])
    path_summary = logs_dir / f"null_test_summary_{ts}.csv"
    df_summary.to_csv(path_summary, index=False)

    path_hits: Path | None = None
    if resultado.n_candidatos > 0:
        estados_np = jax.device_get(resultado.estados)
        errores_np = jax.device_get(resultado.errores)
        df_hits = pd.DataFrame({
            "n1": estados_np[:, 0].astype(int),
            "n2": estados_np[:, 1].astype(int),
            "n3": estados_np[:, 2].astype(int),
            "mse": errores_np,
        })
        path_hits = logs_dir / f"null_test_hits_{ts}.csv"
        df_hits.to_csv(path_hits, index=False)

    return path_summary, path_hits
