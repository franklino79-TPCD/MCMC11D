"""
Punto de entrada: python -m mcmc11d --mode <search|null-test|rg-flow>

Pipeline de auditoria computacional para estabilizacion de moduli
en compactificaciones de variedades de holonomia G2 (Teoria M).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

# ── Logging global antes de cualquier import JAX ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log: Final[logging.Logger] = logging.getLogger("mcmc11d")

# ── JAX init ──────────────────────────────────────────────────────────────
import jax  # noqa: E402

jax.config.update("jax_enable_x64", False)

from mcmc11d.cli.parser import build_parser  # noqa: E402

_SRC_DIR: Path = Path(__file__).resolve().parent
_LOGS_DIR: Path = _SRC_DIR.parent / "data" / "outputs" / "logs"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    print()
    print("+" + "=" * 63 + "+")
    print("|   Auditoria Computacional — Pregeometria M-teorica G2         |")
    print("|   MCMC11D  v2.0  ·  Fases 1-2-3A  [Modular + Holografico]    |")
    print("+" + "=" * 63 + "+")
    log.info("Modo        : %s", args.mode.upper())
    log.info("JAX backend : %s", jax.default_backend())
    log.info("Dispositivos: %s", jax.devices())

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    try:
        if args.mode == "search":
            from mcmc11d.modes.search import ejecutar_search
            ejecutar_search(
                n_max=args.n_max,
                batch_size=args.batch_size,
                epsilon=args.epsilon,
                logs_dir=_LOGS_DIR,
                checkpoint_dir=checkpoint_dir,
                resume=args.resume,
            )

        elif args.mode == "null-test":
            from mcmc11d.modes.null_test_mode import ejecutar_null_test_mode
            ejecutar_null_test_mode(
                n_samples=args.n_samples,
                n_max=args.n_max,
                epsilon=args.epsilon,
                seed=args.seed,
                logs_dir=_LOGS_DIR,
            )

        elif args.mode == "rg-flow":
            from mcmc11d.modes.rg_flow_mode import ejecutar_rg_flow_mode
            ejecutar_rg_flow_mode(
                n_max=args.n_max,
                mu_start=args.mu_start,
                mu_end=args.mu_end,
                mu_steps=args.mu_steps,
                alpha=args.rg_alpha,
                logs_dir=_LOGS_DIR,
            )

    except KeyboardInterrupt:
        print()
        log.warning("Interrumpido por el usuario (Ctrl+C).")
        sys.exit(0)

    except Exception as exc:
        log.error("Error no controlado: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
