"""
Constantes fisicas y de gauge del pipeline MCMC11D.
"""

from __future__ import annotations

import math
from typing import Final

# ── Gauge Fixing: Teorema de Seleccion Topologica del Modo Cero (G2) ────────
N1_GAUGE_FIXED: Final[int] = 1

# ── Constantes fisicas ──────────────────────────────────────────────────────
# Masa de Planck reducida (GeV)
M_PLANCK_REDUCED: Final[float] = 2.435e18

# Limite de Bekenstein para variedad 7D compacta (bits)
# S_B = (2 * pi * R * E) / (hbar * c)
# Para R ~ l_Planck^7 y E ~ M_Planck, la cota superior en bits es:
BEKENSTEIN_BITS_7D: Final[float] = 1e66

# Capacidad computacional maxima del sustrato (ops/s) — derivada del
# limite de Bremermann para volumen de Planck^7
SUSTRATO_OPS_LIMIT: Final[float] = 1e166

# pi en float32
PI_F32: Final[float] = math.pi

# ── Punto fijo clasico esperado ─────────────────────────────────────────────
N2_CLASICO: Final[int] = 207
N3_CLASICO: Final[int] = 17

# ── Defaults del pipeline ───────────────────────────────────────────────────
DEFAULT_N_MAX: Final[int] = 300
DEFAULT_BATCH_SIZE: Final[int] = 100_000
DEFAULT_EPSILON: Final[float] = 1e-3
DEFAULT_SEED: Final[int] = 42
DEFAULT_N_SAMPLES: Final[int] = 1_000_000

# RG-Flow defaults
DEFAULT_MU_START: Final[float] = 10.0
DEFAULT_MU_END: Final[float] = 0.1
DEFAULT_MU_STEPS: Final[int] = 50
DEFAULT_RG_ALPHA: Final[float] = 0.1
