"""
Motor de coste MSE para la auditoria computacional G2-octonionica.

El MSE logaritmico mide la desviacion entre las relaciones de masa
observadas y las predichas por la compactificacion G2 para un triplete
(n1, n2, n3) de numeros cuanticos topologicos.

Relaciones de masa del whitepaper (Tabla 1):
  m_u/m_t, m_c/m_t, m_d/m_b, m_s/m_b, m_e/m_tau, m_mu/m_tau
codificadas como funciones de (n1, n2, n3) via intersecciones de
3-ciclos asociativos en la variedad G2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── Relaciones de masa experimentales (PDG 2024) ───────────────────────────
# log10 de los ratios de masa observados
_TARGETS = jnp.array([
    -5.267,   # log10(m_u / m_t)
    -2.856,   # log10(m_c / m_t)
    -3.371,   # log10(m_d / m_b)
    -1.631,   # log10(m_s / m_b)
    -3.537,   # log10(m_e / m_tau)
    -1.275,   # log10(m_mu / m_tau)
], dtype=jnp.float32)

_N_OBSERVABLES: int = _TARGETS.shape[0]


@jax.jit
def calcular_mse_batch(batch_n: jnp.ndarray) -> jnp.ndarray:
    """MSE logaritmico vectorizado sobre un batch de tripletes.

    Parameters
    ----------
    batch_n : jnp.ndarray shape (B, 3)
        Tripletes (n1, n2, n3) en float32.

    Returns
    -------
    jnp.ndarray shape (B,)
        MSE por estado: mean( (log10_pred_i - log10_obs_i)^2 ).
    """
    n1 = batch_n[:, 0]
    n2 = batch_n[:, 1]
    n3 = batch_n[:, 2]

    eps = jnp.float32(1e-30)

    pred = jnp.stack([
        jnp.log10(jnp.abs(n1 / (n2 * n3)) + eps),
        jnp.log10(jnp.abs(n2 / (n1 * n3)) + eps),
        jnp.log10(jnp.abs(n3 / (n1 * n2)) + eps),
        jnp.log10(jnp.abs((n1 * n2) / (n3 ** 2)) + eps),
        jnp.log10(jnp.abs((n1 * n3) / (n2 ** 2)) + eps),
        jnp.log10(jnp.abs((n2 * n3) / (n1 ** 2)) + eps),
    ], axis=1)  # (B, 6)

    residuos = pred - _TARGETS[None, :]
    return jnp.mean(residuos ** 2, axis=1)


@dataclass(frozen=True)
class ResultadoFiltrado:
    """Resultado del filtrado de candidatos por umbral epsilon."""
    estados: jnp.ndarray     # (K, 3)
    errores: jnp.ndarray     # (K,)
    n_evaluados: int
    n_candidatos: int


def filtrar_candidatos(
    batch: jnp.ndarray,
    epsilon: float = 1e-3,
) -> ResultadoFiltrado:
    """Evalua MSE y filtra estados con MSE < epsilon.

    Ejecuta device_get para la mascara bool, actuando como barrera
    de sincronizacion implicita CPU-GPU.
    """
    mse = calcular_mse_batch(batch)
    mascara = mse < epsilon
    mascara_host = jax.device_get(mascara)

    estados_filtrados = batch[mascara_host]
    errores_filtrados = mse[mascara_host]

    return ResultadoFiltrado(
        estados=estados_filtrados,
        errores=errores_filtrados,
        n_evaluados=int(batch.shape[0]),
        n_candidatos=int(estados_filtrados.shape[0]),
    )
