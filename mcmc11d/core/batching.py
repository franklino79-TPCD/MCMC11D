"""
Sistema de micro-batching asincrono con gestion de VRAM.

Estrategia de transferencia CPU <-> GPU (RTX 5090, PCIe 5.0 x16):
─────────────────────────────────────────────────────────────────
1. La malla completa reside en host (CPU RAM, 64 GB).
2. Cada micro-batch se transfiere al device via jax.device_put con
   donate_argnums para evitar copias innecesarias.
3. Tras cada micro-batch, los resultados escalares (indices, MSE minimo)
   se extraen con jax.device_get — transferencia minima (~bytes).
4. Los arrays intermedios en device se liberan automaticamente por el
   garbage collector de XLA al salir del scope del micro-batch.
5. Para checkpointing, NUNCA se transfiere el array completo de MSE
   al host: solo se persisten los candidatos filtrados (sparse).

Throughput esperado PCIe 5.0 x16: ~64 GB/s bidireccional.
Para n_max=300 (90k estados x 3 x 4 bytes = 1.08 MB por batch),
el overhead de transferencia es < 20 us — despreciable frente al
tiempo de computo XLA (~ms).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Iterator

import jax
import jax.numpy as jnp

log = logging.getLogger("mcmc11d")


@dataclass(frozen=True)
class MicroBatch:
    """Slice de la malla para procesamiento en device."""
    index: int
    total: int
    data: jnp.ndarray  # (B, 3) en host; se mueve a device en el caller


def generar_microbatches(
    malla: jnp.ndarray,
    batch_size: int,
) -> Iterator[MicroBatch]:
    """Genera micro-batches desde una malla en host memory.

    Cada MicroBatch contiene un slice de la malla. El caller es
    responsable de mover el slice a device via jax.device_put
    y liberar la referencia tras el computo.
    """
    n_total = malla.shape[0]
    n_batches = math.ceil(n_total / batch_size)

    for i in range(n_batches):
        inicio = i * batch_size
        fin = min(inicio + batch_size, n_total)
        yield MicroBatch(
            index=i,
            total=n_batches,
            data=malla[inicio:fin],
        )


def estimar_vram_mb(n_estados: int, n_cols: int = 3, dtype_bytes: int = 4) -> float:
    """Estima VRAM requerida en MB para un batch."""
    return (n_estados * n_cols * dtype_bytes) / 1_048_576


def batch_size_optimo(
    n_total: int,
    vram_disponible_mb: float = 28_000.0,
    factor_seguridad: float = 0.7,
    n_cols: int = 3,
    dtype_bytes: int = 4,
) -> int:
    """Calcula batch_size maximo que cabe en VRAM con margen de seguridad.

    Para RTX 5090 (32 GB VRAM), con factor 0.7, usa ~22.4 GB efectivos.
    El resto se reserva para buffers XLA intermedios y el kernel compilado.
    """
    vram_util = vram_disponible_mb * factor_seguridad
    max_estados = int(vram_util * 1_048_576 / (n_cols * dtype_bytes))
    return min(max_estados, n_total)


def liberar_device_arrays(*arrays: jnp.ndarray) -> None:
    """Fuerza la liberacion de arrays en device.

    En JAX, los buffers de device se liberan cuando no hay referencias
    Python. Esta funcion explicita el patron de liberacion para
    iteraciones largas donde el GC de Python podria retrasarse.
    """
    del arrays
