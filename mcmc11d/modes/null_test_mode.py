"""
Orquestador del modo NULL-TEST — validacion Monte Carlo.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from mcmc11d.analysis.null_test import ejecutar_null_test, exportar_resultados

log = logging.getLogger("mcmc11d")


def ejecutar_null_test_mode(
    n_samples: int,
    n_max: int,
    epsilon: float,
    seed: int,
    logs_dir: Path,
) -> None:
    log.info("Modo NULL-TEST iniciado")
    log.info("  n_samples = %s", f"{n_samples:,}")
    log.info("  rango     = [1, %d]", n_max)
    log.info("  epsilon   = %.0e", epsilon)
    log.info("  seed      = %d", seed)

    t0 = time.perf_counter()
    resultado, p_value = ejecutar_null_test(
        n_samples=n_samples, n_min=1, n_max=n_max,
        epsilon=epsilon, seed=seed,
    )
    dt = time.perf_counter() - t0
    throughput = n_samples / dt / 1e6

    print()
    print("=" * 65)
    print("  RESULTADOS — MODO NULL-TEST")
    print("=" * 65)
    print(f"  Muestras evaluadas  : {resultado.n_evaluados:,}")
    print(f"  Hits aleatorios     : {resultado.n_candidatos:,}")
    print(f"  P-valor             : {p_value:.6e}")
    print(f"  Throughput          : {throughput:.2f} M estados/s")
    print(f"  Tiempo total        : {dt:.2f} s")
    print("-" * 65)

    if p_value < 0.01:
        veredicto = "SIGNIFICATIVO  (p < 0.01)"
        detalle = "Los candidatos sistematicos son estadisticamente no-aleatorios."
    elif p_value < 0.05:
        veredicto = "MARGINAL  (0.01 <= p < 0.05)"
        detalle = "Considera reducir epsilon o aumentar n_samples."
    else:
        veredicto = "NO SIGNIFICATIVO  (p >= 0.05)"
        detalle = "Epsilon demasiado permisivo; ajustar antes de publicar."

    print(f"  Veredicto : {veredicto}")
    print(f"  Detalle   : {detalle}")
    print("=" * 65)

    path_summary, path_hits = exportar_resultados(
        resultado=resultado, p_value=p_value,
        n_min=1, n_max=n_max, epsilon=epsilon,
        seed=seed, logs_dir=logs_dir,
    )

    print(f"\n  Resumen -> {path_summary.name}")
    if path_hits is not None:
        print(f"  Hits    -> {path_hits.name}  ({resultado.n_candidatos} filas)")
    else:
        print("  Hits    -> (sin archivo; n_hits = 0)")
    print(f"  Dir     -> {logs_dir}\n")

    log.info("Modo NULL-TEST completado. p-valor = %.6e", p_value)
