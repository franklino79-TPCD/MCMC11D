"""
Modelo de Inferencia Holografica — Limite de Bekenstein 7D.

Implementa la condicion de parada holografica para el flujo RG:
el atractor se estabiliza cuando la entropia del sistema simulado
alcanza la cota de Bekenstein para una variedad compacta 7D.

Marco teorico:
─────────────
La conjetura holografica (t'Hooft-Susskind) establece que la maxima
informacion codificable en una region de volumen V esta acotada por
el area de su frontera en unidades de Planck:

    S_max = A / (4 * l_P^2)

Para una variedad G2 compacta 7D de radio R, la frontera holografica
es una 6-esfera, y la cota de Bekenstein generalizada se expresa como:

    S_B^{7D} = (2 * pi * R_7 * E) / (hbar * c)

donde R_7 es el radio efectivo de la variedad compacta y E la energia
total del sistema.

La entropia del flujo RG se define como la entropia de Shannon del
landscape de MSE normalizado:

    S_RG(mu) = -sum_i [ p_i(mu) * ln(p_i(mu)) ]

con p_i(mu) = exp(-MSE_eff_i / T) / Z(mu) la distribucion de Boltzmann
sobre el espacio de estados gauge-fijo a temperatura efectiva T ~ mu.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from mcmc11d.core.constants import BEKENSTEIN_BITS_7D


@jax.jit
def entropia_shannon_landscape(
    mse_array: jnp.ndarray,
    temperatura: jnp.ndarray,
) -> jnp.ndarray:
    """Entropia de Shannon del landscape de MSE a temperatura efectiva T.

    S = -sum_i p_i * ln(p_i) donde p_i = softmax(-MSE_i / T).

    La temperatura T ~ mu mapea la escala de energia del flujo RG
    a una temperatura termodinamica efectiva del landscape.
    """
    log_boltzmann = -mse_array / (temperatura + jnp.float32(1e-30))

    log_Z = jax.nn.logsumexp(log_boltzmann)
    log_p = log_boltzmann - log_Z
    p = jnp.exp(log_p)

    entropia = -jnp.sum(p * log_p)
    return entropia


@jax.jit
def ratio_bekenstein(
    mse_array: jnp.ndarray,
    temperatura: jnp.ndarray,
    s_bekenstein: jnp.ndarray,
) -> jnp.ndarray:
    """Ratio S_RG / S_Bekenstein.

    Cuando este ratio alcanza 1.0, el sistema ha saturado la capacidad
    informacional del sustrato holografico y el flujo RG debe detenerse.
    """
    s_rg = entropia_shannon_landscape(mse_array, temperatura)
    return s_rg / s_bekenstein


def umbral_fidelidad_perceptual(
    n_estados: int,
    mu: float,
) -> float:
    """Calcula epsilon_holografico: el umbral de fidelidad perceptual.

    Este epsilon representa la resolucion minima con la que el sustrato
    de renderizado puede distinguir estados del landscape. Se deriva
    del limite de Bekenstein:

        epsilon_holo = ln(N_estados) / S_B^{7D}

    donde N_estados es la cardinalidad del espacio gauge-fijo y S_B^{7D}
    la cota de Bekenstein para la variedad G2 compacta.

    Cuando epsilon_holo >= epsilon_clasico, el sustrato no puede
    resolver mas estructura en el landscape y el vacio queda fijado.
    """
    if n_estados <= 1:
        return 0.0
    return math.log(n_estados) / BEKENSTEIN_BITS_7D


@jax.jit
def condicion_parada_holografica(
    mse_array: jnp.ndarray,
    temperatura: jnp.ndarray,
    s_bekenstein: jnp.ndarray,
    beta_n2: jnp.ndarray,
    beta_n3: jnp.ndarray,
    umbral_beta: jnp.ndarray,
) -> jnp.ndarray:
    """Condicion de parada compuesta para el flujo RG.

    El flujo se detiene (retorna True) cuando se cumple CUALQUIERA de:

    1. Saturacion holografica: S_RG >= S_Bekenstein
       (el sustrato agoto su capacidad informacional)

    2. Punto fijo clasico: |beta_n2| < umbral AND |beta_n3| < umbral
       (las funciones beta se anulan — invariancia de escala)

    Ambas condiciones son equivalentes en el limite termodinamico:
    la saturacion de Bekenstein implica que el sistema no puede
    acceder a estados adicionales del landscape, lo cual fuerza
    beta -> 0 por agotamiento del espacio de fases accesible.
    """
    s_rg = entropia_shannon_landscape(mse_array, temperatura)
    saturado = s_rg >= s_bekenstein

    betas_nulas = (jnp.abs(beta_n2) < umbral_beta) & (jnp.abs(beta_n3) < umbral_beta)

    return saturado | betas_nulas
