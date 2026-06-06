"""
CLI argument parser para MCMC11D.
"""

from __future__ import annotations

import argparse

from mcmc11d.core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPSILON,
    DEFAULT_MU_END,
    DEFAULT_MU_START,
    DEFAULT_MU_STEPS,
    DEFAULT_N_MAX,
    DEFAULT_N_SAMPLES,
    DEFAULT_RG_ALPHA,
    DEFAULT_SEED,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcmc11d",
        description="Auditoria computacional G2-Octonionica — Fases 1, 2 y 3A",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["search", "null-test", "rg-flow"],
        required=True,
        help="'search' barrido sistematico; 'null-test' Monte Carlo; 'rg-flow' Flujo RG.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON, metavar="FLOAT",
        help="Umbral de fidelidad perceptual (epsilon holografico) para filtrar candidatos.",
    )
    parser.add_argument(
        "--n-max", type=int, default=DEFAULT_N_MAX, dest="n_max", metavar="INT",
        help="Limite superior del espacio gauge-fijo {1} x [1, n_max]^2.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, dest="batch_size", metavar="INT",
        help="Estados por micro-batch (modo search).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, metavar="INT",
        help="Semilla PRNG para el null-test Monte Carlo.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES, dest="n_samples", metavar="INT",
        help="Tripletes aleatorios a evaluar en modo null-test.",
    )
    # ── RG-flow ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--mu-start", type=float, default=DEFAULT_MU_START, dest="mu_start", metavar="FLOAT",
        help="[rg-flow] Escala UV inicial.",
    )
    parser.add_argument(
        "--mu-end", type=float, default=DEFAULT_MU_END, dest="mu_end", metavar="FLOAT",
        help="[rg-flow] Escala IR final.",
    )
    parser.add_argument(
        "--mu-steps", type=int, default=DEFAULT_MU_STEPS, dest="mu_steps", metavar="INT",
        help="[rg-flow] Escalas de energia intermedias.",
    )
    parser.add_argument(
        "--rg-alpha", type=float, default=DEFAULT_RG_ALPHA, dest="rg_alpha", metavar="FLOAT",
        help="[rg-flow] Coeficiente alpha de correccion logaritmica.",
    )
    # ── General ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, dest="checkpoint_dir", metavar="PATH",
        help="Directorio para checkpoints (habilita reanudacion tolerante a fallos).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reanudar desde el ultimo checkpoint valido.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Activar logging DEBUG.",
    )

    return parser
