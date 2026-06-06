"""
Construccion de mallas gauge-fijas sobre la variedad G2.

La malla se construye en el espacio gauge-fijo {n1=1} x [1, n_max]^2,
explotando que n1 es un modo cero del operador de Laplace-Beltrami
en la variedad G2 (no contribuye al MSE ni a ningun observable fisico).
"""

from __future__ import annotations

import jax.numpy as jnp

from mcmc11d.core.constants import N1_GAUGE_FIXED


def construir_malla_gauge_fija(n_max: int) -> jnp.ndarray:
    """Malla gauge-fija {1} x [1, n_max]^2 como array JAX (n_max^2, 3) float32."""
    n_total = n_max ** 2
    idx = jnp.arange(1, n_max + 1, dtype=jnp.int32)
    n2, n3 = jnp.meshgrid(idx, idx, indexing="ij")
    n1_col = jnp.full(n_total, N1_GAUGE_FIXED, dtype=jnp.int32)
    return jnp.stack(
        [n1_col, n2.reshape(-1), n3.reshape(-1)],
        axis=1,
    ).astype(jnp.float32)


def construir_malla_global(n_max: int) -> jnp.ndarray:
    """Malla completa [1, n_max]^3 — solo para null-test o validacion cruzada."""
    idx = jnp.arange(1, n_max + 1, dtype=jnp.int32)
    n1, n2, n3 = jnp.meshgrid(idx, idx, idx, indexing="ij")
    return jnp.stack(
        [n1.reshape(-1), n2.reshape(-1), n3.reshape(-1)],
        axis=1,
    ).astype(jnp.float32)
