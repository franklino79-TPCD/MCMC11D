"""CKM matrix from G2 holonomy 3-cycle intersection geometry.

Computes quark flavor mixing angles (CKM matrix elements) from the
intersection geometry of associative 3-cycles in a G2 holonomy manifold
constructed via Twisted Connected Sum (TCS).

Geometric Ansatz (TCS asymptotic neck limit):
    In TCS geometry, flavor transition amplitudes are determined by
    intersection phases between codimension-7 singularities. The mixing
    angles are logarithmically suppressed by the stabilized cycle volumes:

    - Cabibbo angle: theta_c ~ 1 / ln(n2 / n1)
      The dominant 1-2 generation mixing is set by the logarithmic ratio
      of the first two cycle volumes. This encodes the geometric distance
      between the singularity loci supporting the first two quark generations
      in the TCS neck region.

    - V_cb ~ V_us^2 / ln(n3)
      The 2-3 mixing is suppressed by the square of the Cabibbo angle and
      an additional logarithmic factor from the heavy cycle volume n3,
      reflecting the deeper embedding of the third generation singularity
      in the bulk of the G2 manifold.

    - V_ub ~ V_us^3 * (n1 / n3)
      The 1-3 mixing carries cubic Cabibbo suppression plus a direct
      geometric ratio n1/n3, encoding the maximal separation between
      the lightest and heaviest generation loci on the TCS building blocks.

PDG 2024 experimental targets:
    |V_us| = 0.2250 +/- 0.0005
    |V_cb| = 0.0413 +/- 0.0012
    |V_ub| = 0.00379 +/- 0.00020
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── Experimental CKM magnitudes (PDG 2024) ─────────────────────────────────

_V_US_EXP: jnp.ndarray = jnp.float32(0.2250)
_V_CB_EXP: jnp.ndarray = jnp.float32(0.0413)
_V_UB_EXP: jnp.ndarray = jnp.float32(0.00379)

_TARGETS_CKM: jnp.ndarray = jnp.array(
    [0.2250, 0.0413, 0.00379], dtype=jnp.float32,
)


class CKMResult(NamedTuple):
    """Result container for geometric CKM computation.

    Attributes:
        matrix: (3, 3) real magnitude matrix |V_ij|.
        v_us: |V_us| = sin(theta_c), Cabibbo element.
        v_cb: |V_cb|, 2-3 generation mixing.
        v_ub: |V_ub|, 1-3 generation mixing.
        mse: Mean squared error vs PDG 2024 targets.
    """
    matrix: jnp.ndarray
    v_us: jnp.ndarray
    v_cb: jnp.ndarray
    v_ub: jnp.ndarray
    mse: jnp.ndarray


@jax.jit
def calcular_matriz_ckm_geometrica(
    n1: jnp.ndarray,
    n2: jnp.ndarray,
    n3: jnp.ndarray,
    scale_factor: jnp.ndarray = jnp.float32(1.0),
) -> CKMResult:
    """Computes the CKM matrix from TCS 3-cycle intersection geometry.

    The function is pure and JIT-compiled. All operations use jax.numpy
    to remain inside the XLA tracer, enabling future vmap over full
    gauge-fixed meshes without retracing.

    Args:
        n1: First cycle quantum number (gauge-fixed to 1 in production).
        n2: Second cycle quantum number (free, attractor at 207).
        n3: Third cycle quantum number (free, attractor at 17).
        scale_factor: Overall rescaling of the Cabibbo angle. Defaults
            to 1.0 (no rescaling). Useful for RG-flow studies where the
            effective coupling runs with the energy scale mu.

    Returns:
        CKMResult with the 3x3 magnitude matrix, individual elements,
        and MSE against PDG 2024.
    """
    # ── Cabibbo angle from 1-2 cycle volume ratio ───────────────────────
    # theta_c = scale_factor / ln(n2 / n1)
    # For the attractor (1, 207, 17): theta_c ~ 1/ln(207) ~ 0.1873
    # sin(theta_c) ~ 0.186 ... close to 0.225 with scale tuning.
    ratio_12 = jnp.abs(n2) / (jnp.abs(n1) + jnp.float32(1e-30))
    ln_ratio = jnp.log(ratio_12 + jnp.float32(1e-30))
    theta_c = scale_factor / (ln_ratio + jnp.float32(1e-30))

    v_us = jnp.sin(theta_c)

    # ── 2-3 mixing: quadratic Cabibbo suppression + heavy cycle ─────────
    # V_cb = V_us^2 / ln(n3)
    ln_n3 = jnp.log(jnp.abs(n3) + jnp.float32(1e-30))
    v_cb = (v_us ** 2) / (ln_n3 + jnp.float32(1e-30))

    # ── 1-3 mixing: cubic Cabibbo + geometric ratio ─────────────────────
    # V_ub = V_us^3 * (n1 / n3)
    v_ub = (v_us ** 3) * (jnp.abs(n1) / (jnp.abs(n3) + jnp.float32(1e-30)))

    # ── Unitarity-constrained 3x3 magnitude matrix ──────────────────────
    # Wolfenstein-like parametrization (real magnitudes, CP phase ignored):
    #   |V| = [[V_ud,  V_us,  V_ub ],
    #          [V_cd,  V_cs,  V_cb ],
    #          [V_td,  V_ts,  V_tb ]]
    #
    # Row/column unitarity:
    #   |V_ud|^2 + |V_us|^2 + |V_ub|^2 = 1
    #   |V_cd|^2 + |V_cs|^2 + |V_cb|^2 = 1
    #   |V_td|^2 + |V_ts|^2 + |V_tb|^2 = 1
    v_ud = jnp.sqrt(jnp.maximum(
        jnp.float32(1.0) - v_us ** 2 - v_ub ** 2,
        jnp.float32(0.0),
    ))
    v_cs = jnp.sqrt(jnp.maximum(
        jnp.float32(1.0) - v_us ** 2 - v_cb ** 2,
        jnp.float32(0.0),
    ))
    v_tb = jnp.sqrt(jnp.maximum(
        jnp.float32(1.0) - v_ub ** 2 - v_cb ** 2,
        jnp.float32(0.0),
    ))

    # Off-diagonal elements from unitarity (leading order Wolfenstein)
    v_cd = v_us
    v_td = v_ub * v_cs - v_us * v_cb
    v_td = jnp.abs(v_td)
    v_ts = v_cb

    matrix = jnp.array([
        [v_ud, v_us, v_ub],
        [v_cd, v_cs, v_cb],
        [v_td, v_ts, v_tb],
    ])

    # ── MSE vs PDG 2024 ────────────────────────────────────────────────
    pred = jnp.array([v_us, v_cb, v_ub])
    mse = jnp.mean((pred - _TARGETS_CKM) ** 2)

    return CKMResult(
        matrix=matrix,
        v_us=v_us,
        v_cb=v_cb,
        v_ub=v_ub,
        mse=mse,
    )


@jax.jit
def calcular_mse_ckm_batch(batch_n: jnp.ndarray) -> jnp.ndarray:
    """Vectorized CKM MSE over a batch of triplets.

    Designed for direct injection into the gauge-fixed mesh sweep.
    Uses vmap internally so the caller passes the full (B, 3) mesh
    without manual batching.

    Args:
        batch_n: (B, 3) float32 array of (n1, n2, n3) triplets.

    Returns:
        (B,) float32 array of CKM MSE per state.
    """
    def _single_mse(row: jnp.ndarray) -> jnp.ndarray:
        result = calcular_matriz_ckm_geometrica(row[0], row[1], row[2])
        return result.mse

    return jax.vmap(_single_mse)(batch_n)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)

    print("=" * 65)
    print("  CKM Intersection Geometry — Sanity Check")
    print("=" * 65)

    # ── Attractor point (1, 207, 17) ────────────────────────────────────
    r_att = calcular_matriz_ckm_geometrica(
        jnp.float32(1.0), jnp.float32(207.0), jnp.float32(17.0),
    )
    print("\n  Attractor (1, 207, 17):")
    print(f"    |V_us| = {float(r_att.v_us):.6f}   (exp: 0.2250)")
    print(f"    |V_cb| = {float(r_att.v_cb):.6f}   (exp: 0.0413)")
    print(f"    |V_ub| = {float(r_att.v_ub):.6f}  (exp: 0.00379)")
    print(f"    MSE    = {float(r_att.mse):.6e}")
    print(f"    |V| matrix:\n{jax.device_get(r_att.matrix)}")

    # ── Noise point (1, 50, 50) ─────────────────────────────────────────
    r_noise = calcular_matriz_ckm_geometrica(
        jnp.float32(1.0), jnp.float32(50.0), jnp.float32(50.0),
    )
    print("\n  Noise (1, 50, 50):")
    print(f"    |V_us| = {float(r_noise.v_us):.6f}   (exp: 0.2250)")
    print(f"    |V_cb| = {float(r_noise.v_cb):.6f}   (exp: 0.0413)")
    print(f"    |V_ub| = {float(r_noise.v_ub):.6f}  (exp: 0.00379)")
    print(f"    MSE    = {float(r_noise.mse):.6e}")

    # ── Batch test (vmap) ───────────────────────────────────────────────
    batch = jnp.array([
        [1.0, 207.0, 17.0],
        [1.0, 50.0, 50.0],
        [1.0, 100.0, 30.0],
        [1.0, 300.0, 10.0],
    ])
    mse_batch = calcular_mse_ckm_batch(batch)
    print("\n  Batch CKM MSE (vmap):")
    for i, row in enumerate(jax.device_get(batch)):
        print(f"    ({int(row[0])}, {int(row[1])}, {int(row[2])})  ->  MSE = {float(mse_batch[i]):.6e}")

    print()
    mse_ratio = float(r_noise.mse) / float(r_att.mse)
    print(f"  MSE ratio noise/attractor: {mse_ratio:.1f}x")
    if float(r_att.mse) < float(r_noise.mse):
        print("  -> Attractor (207,17) produces LOWER CKM error than noise.")
    print("=" * 65)
