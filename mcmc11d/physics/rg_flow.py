"""
Flujo del Grupo de Renormalizacion (Fase 3A) con parada holografica.

Este modulo implementa el barrido de la escala de energia running mu
desde el regimen UV (alta energia) hasta el regimen IR (baja energia),
trazando la trayectoria del minimo absoluto de MSE_eff(n, mu).

Potencial efectivo:
───────────────────
    MSE_eff(n, mu) = MSE_clasico(n) * (1 + alpha * ln(mu))
                     + delta_q(n, mu)

Correccion cuantica (instantones en variedad G2):
    delta_q(n, mu) = (1/8) * sum_k [ sin^2(pi * n_k * mu) / (n_k^2 + mu^2) ]

Condicion de parada holografica:
────────────────────────────────
El flujo RG se detiene cuando la entropia de Shannon del landscape
de MSE_eff a temperatura T ~ mu satura el limite de Bekenstein para
la variedad G2 compacta 7D. Esto equivale fisicamente a que el
sustrato computacional no puede resolver mas estructura fina en el
landscape: el vacio queda fijado por agotamiento informacional.

Nota sobre la compilacion XLA:
─────────────────────────────
Las funciones marcadas con @jax.jit se compilan a kernels XLA/HLO
que se ejecutan como grafos estaticos en el acelerador. La semantica
de evaluacion diferida (lazy evaluation) del trazador XLA implica
que las operaciones solo se materializan cuando un resultado concreto
se solicita (via .block_until_ready() o jax.device_get). Este patron
de evaluacion perezosa es isomorfo al principio de economia computacional:
el grafo XLA solo computa los tensores que contribuyen al observable
solicitado, descartando ramas muertas del grafo en tiempo de compilacion.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Final

import jax
import jax.numpy as jnp

from mcmc11d.core.constants import (
    BEKENSTEIN_BITS_7D,
    N2_CLASICO,
    N3_CLASICO,
    PI_F32,
)
from mcmc11d.core.mesh import construir_malla_gauge_fija
from mcmc11d.physics.cost_engine import calcular_mse_batch
from mcmc11d.physics.holographic import (
    condicion_parada_holografica,
    entropia_shannon_landscape,
    umbral_fidelidad_perceptual,
)

log = logging.getLogger("mcmc11d")


# ── MSE efectivo parametrizado por escala mu ────────────────────────────────

@jax.jit
def calcular_mse_efectivo(
    batch_n: jnp.ndarray,
    mu: jnp.ndarray,
    alpha: jnp.ndarray,
) -> jnp.ndarray:
    """MSE efectivo parametrizado por escala de energia running mu.

    El kernel XLA compilado por este JIT funde las operaciones de MSE clasico,
    factor de escala logaritmico y correccion cuantica en un unico grafo HLO.
    La evaluacion diferida del trazador XLA garantiza que solo se materializan
    los tensores que contribuyen al output final — las ramas intermedias que
    no afectan al resultado se eliminan en la pasada de optimizacion DCE
    (Dead Code Elimination) del compilador XLA, minimizando el consumo de
    VRAM y ciclos de computo en el acelerador.

    Parameters
    ----------
    batch_n : (B, 3) float32 — tripletes gauge-fijos
    mu : scalar float32 — escala de energia running
    alpha : scalar float32 — coeficiente de correccion logaritmica

    Returns
    -------
    (B,) float32 — MSE efectivo por estado
    """
    mse_clasico = calcular_mse_batch(batch_n)

    # Factor de escala logaritmico — log1p para estabilidad en mu -> 0
    escala_log = jnp.float32(1.0) + alpha * jnp.log1p(jnp.abs(mu))

    # Correccion cuantica tipo instanton
    mu_sq = mu * mu + jnp.float32(1e-12)
    pi_mu = jnp.float32(PI_F32) * mu

    def _contrib(n_col: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(pi_mu * n_col) ** 2 / (n_col ** 2 + mu_sq)

    delta_q = (
        _contrib(batch_n[:, 0])
        + _contrib(batch_n[:, 1])
        + _contrib(batch_n[:, 2])
    ) / jnp.float32(8.0)

    return mse_clasico * escala_log + delta_q


# ── Registro de un paso RG ──────────────────────────────────────────────────

@dataclass
class RegistroRG:
    """Snapshot de un paso del flujo RG."""
    mu: float
    ln_mu: float
    top_n2: int
    top_n3: int
    min_mse: float
    beta_n2: float
    beta_n3: float
    convergido: bool
    entropia_rg: float
    ratio_bekenstein: float
    parada_holografica: bool


@dataclass
class ResultadoRGFlow:
    """Resultado completo del barrido RG."""
    registros: list[RegistroRG] = field(default_factory=list)
    n2_ir_final: int = 0
    n3_ir_final: int = 0
    beta_n2_mean_ir: float = float("nan")
    beta_n3_mean_ir: float = float("nan")
    punto_fijo_207_17: bool = False
    atractor_confirmado: bool = False
    parada_por_bekenstein: bool = False
    veredicto: str = ""
    tiempo_total_s: float = 0.0


def ejecutar_rg_flow(
    n_max: int,
    mu_start: float,
    mu_end: float,
    mu_steps: int,
    alpha: float,
    umbral_beta: float = 0.5,
) -> ResultadoRGFlow:
    """Ejecuta el barrido completo del flujo RG con parada holografica.

    Para cada escala mu en [mu_UV, mu_IR] (espacio logaritmico):
    1. Evalua MSE_eff sobre la malla gauge-fija completa.
    2. Localiza el minimo absoluto (n2*, n3*).
    3. Calcula funciones beta via diferencias finitas en ln(mu).
    4. Evalua la entropia de Shannon del landscape.
    5. Verifica la condicion de parada holografica (Bekenstein).

    La compilacion JIT de calcular_mse_efectivo fusiona todo el pipeline
    de evaluacion en un unico kernel XLA. El trazador diferido solo
    materializa el tensor de MSE_eff cuando block_until_ready() fuerza
    la sincronizacion — el acelerador evalua el grafo completo en una
    unica pasada, sin roundtrips CPU-GPU intermedios.
    """
    n_total = n_max ** 2

    log.info("Modo RG-FLOW iniciado  [Fase 3A + Parada Holografica]")
    log.info("  Escala UV (mu_start) : %.3f", mu_start)
    log.info("  Escala IR (mu_end)   : %.3f", mu_end)
    log.info("  Pasos mu             : %d", mu_steps)
    log.info("  Alpha                : %.4f", alpha)
    log.info("  Malla gauge-fija     : {1} x [1, %d]^2 = %s estados", n_max, f"{n_total:,}")

    # ── Espacio de escalas logaritmico UV -> IR ────────────────────────────
    log_mu_arr = jnp.linspace(
        jnp.log(jnp.float32(mu_start)),
        jnp.log(jnp.float32(mu_end)),
        mu_steps,
    )
    mu_arr = jnp.exp(log_mu_arr)

    # ── Malla gauge-fija ──────────────────────────────────────────────────
    log.info("Construyendo malla gauge-fija...")
    t0 = time.perf_counter()
    malla = construir_malla_gauge_fija(n_max)
    malla.block_until_ready()
    log.info("  Malla lista: shape=%s (%.0f ms)",
             malla.shape, 1000 * (time.perf_counter() - t0))

    # ── Warmup JIT ────────────────────────────────────────────────────────
    log.info("Compilando kernel MSE_eff (warmup JIT)...")
    _mu0 = jnp.float32(mu_arr[0])
    _alpha = jnp.float32(alpha)
    _ = calcular_mse_efectivo(malla[:10], _mu0, _alpha).block_until_ready()
    log.info("  Kernel compilado.")

    # ── Limite de Bekenstein para la variedad G2 compacta ─────────────────
    s_bekenstein = jnp.float32(BEKENSTEIN_BITS_7D)
    epsilon_holo = umbral_fidelidad_perceptual(n_total, mu_start)
    log.info("  S_Bekenstein^{7D}    : %.2e bits", BEKENSTEIN_BITS_7D)
    log.info("  epsilon_holografico  : %.4e", epsilon_holo)

    # ── Barrido principal ─────────────────────────────────────────────────
    registros: list[RegistroRG] = []
    n2_prev: float | None = None
    n3_prev: float | None = None
    lnmu_prev: float | None = None
    parada_por_bekenstein = False

    print()
    print("=" * 78)
    print("  BARRIDO RG-FLOW  —  mu: {:.3f} (UV)  ->  {:.3f} (IR)  [Bekenstein activo]".format(
        mu_start, mu_end))
    print("=" * 78)
    print(f"  {'paso':>4}  {'mu':>8}  {'ln(mu)':>8}  {'n2*':>6}  {'n3*':>6}  "
          f"{'MSE_min':>12}  {'beta_n2':>9}  {'beta_n3':>9}  {'S/S_B':>8}  {'estado':>6}")
    print("  " + "-" * 74)

    t_sweep = time.perf_counter()

    for step, (mu_val, lnmu_val) in enumerate(
        zip(jax.device_get(mu_arr), jax.device_get(log_mu_arr))
    ):
        mu_jnp = jnp.float32(mu_val)
        alpha_jnp = jnp.float32(alpha)

        mse_eff = calcular_mse_efectivo(malla, mu_jnp, alpha_jnp)
        mse_eff.block_until_ready()

        idx_min = int(jnp.argmin(mse_eff))
        min_mse = float(mse_eff[idx_min])
        n2_star = int(malla[idx_min, 1])
        n3_star = int(malla[idx_min, 2])
        lnmu_val = float(lnmu_val)

        # Funciones beta
        if n2_prev is not None and lnmu_prev is not None:
            delta_lnmu = lnmu_val - lnmu_prev
            beta_n2 = (n2_star - n2_prev) / delta_lnmu if abs(delta_lnmu) > 1e-12 else 0.0
            beta_n3 = (n3_star - n3_prev) / delta_lnmu if abs(delta_lnmu) > 1e-12 else 0.0
        else:
            beta_n2 = float("nan")
            beta_n3 = float("nan")

        # Entropia de Shannon del landscape a T ~ mu
        temperatura = jnp.float32(float(mu_val))
        s_rg = float(entropia_shannon_landscape(mse_eff, temperatura))
        ratio_bek = s_rg / BEKENSTEIN_BITS_7D

        # Condicion de parada holografica
        convergido_clasico = (
            not math.isnan(beta_n2) and not math.isnan(beta_n3)
            and abs(beta_n2) < umbral_beta and abs(beta_n3) < umbral_beta
        )

        parada_holo = s_rg >= BEKENSTEIN_BITS_7D

        if convergido_clasico:
            estado_str = "FIJO"
        elif parada_holo:
            estado_str = "BEKEN"
            parada_por_bekenstein = True
        else:
            estado_str = "  "

        registros.append(RegistroRG(
            mu=float(mu_val),
            ln_mu=lnmu_val,
            top_n2=n2_star,
            top_n3=n3_star,
            min_mse=min_mse,
            beta_n2=beta_n2,
            beta_n3=beta_n3,
            convergido=convergido_clasico,
            entropia_rg=s_rg,
            ratio_bekenstein=ratio_bek,
            parada_holografica=parada_holo,
        ))

        n2_prev = n2_star
        n3_prev = n3_star
        lnmu_prev = lnmu_val

        beta_n2_str = f"{beta_n2:>9.3f}" if not math.isnan(beta_n2) else "      nan"
        beta_n3_str = f"{beta_n3:>9.3f}" if not math.isnan(beta_n3) else "      nan"

        print(
            f"  {step+1:>4}  {float(mu_val):>8.4f}  {lnmu_val:>8.4f}  "
            f"{n2_star:>6}  {n3_star:>6}  {min_mse:>12.6e}  "
            f"{beta_n2_str}  {beta_n3_str}  {ratio_bek:>8.2e}  {estado_str:>6}",
            flush=True,
        )

        if parada_holo:
            log.info("Parada holografica alcanzada en step %d (mu=%.4f)", step, float(mu_val))
            break

    dt_total = time.perf_counter() - t_sweep

    # ── Analisis IR ───────────────────────────────────────────────────────
    n_ir = max(1, len(registros) // 5)
    registros_ir = registros[-n_ir:]

    n2_ir_vals = [r.top_n2 for r in registros_ir]
    n3_ir_vals = [r.top_n3 for r in registros_ir]
    beta_ir_n2 = [r.beta_n2 for r in registros_ir if not math.isnan(r.beta_n2)]
    beta_ir_n3 = [r.beta_n3 for r in registros_ir if not math.isnan(r.beta_n3)]

    n2_ir_unique = len(set(n2_ir_vals))
    n3_ir_unique = len(set(n3_ir_vals))
    n2_ir_final = registros[-1].top_n2
    n3_ir_final = registros[-1].top_n3

    beta_n2_mean_ir = sum(abs(b) for b in beta_ir_n2) / len(beta_ir_n2) if beta_ir_n2 else float("nan")
    beta_n3_mean_ir = sum(abs(b) for b in beta_ir_n3) / len(beta_ir_n3) if beta_ir_n3 else float("nan")

    betas_convergen = (
        not math.isnan(beta_n2_mean_ir) and not math.isnan(beta_n3_mean_ir)
        and beta_n2_mean_ir < umbral_beta and beta_n3_mean_ir < umbral_beta
    )
    punto_fijo_207_17 = (n2_ir_final == N2_CLASICO and n3_ir_final == N3_CLASICO)
    atractor_confirmado = betas_convergen and n2_ir_unique == 1 and n3_ir_unique == 1

    if parada_por_bekenstein:
        veredicto = (
            f"BEKENSTEIN — Parada holografica: S_RG satura S_B^{{7D}} "
            f"en ({n2_ir_final}, {n3_ir_final})"
        )
    elif atractor_confirmado and punto_fijo_207_17:
        veredicto = "ATRACTOR TOPOLOGICO CONFIRMADO — (207, 17) es punto fijo RG estable"
    elif atractor_confirmado:
        veredicto = f"ATRACTOR CONFIRMADO en ({n2_ir_final}, {n3_ir_final}) — difiere del clasico"
    elif betas_convergen:
        veredicto = "CONVERGENCIA PARCIAL — betas -> 0 pero trayectoria inestable"
    else:
        veredicto = "NO CONVERGENTE — aumentar mu_steps o revisar alpha"

    resultado = ResultadoRGFlow(
        registros=registros,
        n2_ir_final=n2_ir_final,
        n3_ir_final=n3_ir_final,
        beta_n2_mean_ir=beta_n2_mean_ir,
        beta_n3_mean_ir=beta_n3_mean_ir,
        punto_fijo_207_17=punto_fijo_207_17,
        atractor_confirmado=atractor_confirmado,
        parada_por_bekenstein=parada_por_bekenstein,
        veredicto=veredicto,
        tiempo_total_s=dt_total,
    )

    # ── Reporte consola ───────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  REPORTE FINAL — RG-FLOW")
    print("=" * 78)
    print(f"  Tiempo total             : {dt_total:.2f} s")
    print(f"  Pasos ejecutados         : {len(registros)}/{mu_steps}")
    print(f"  Minimo IR final          : (n2={n2_ir_final}, n3={n3_ir_final})")
    print(f"  Punto esperado (207,17)  : {'COINCIDE' if punto_fijo_207_17 else 'NO COINCIDE'}")
    print(f"  |beta_n2| medio IR       : {beta_n2_mean_ir:.4f}")
    print(f"  |beta_n3| medio IR       : {beta_n3_mean_ir:.4f}")
    print(f"  Parada por Bekenstein    : {'SI' if parada_por_bekenstein else 'NO'}")
    print(f"  VEREDICTO : {veredicto}")
    print("=" * 78)

    return resultado
