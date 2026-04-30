"""Dataset schema helpers for efficiency prediction."""

from typing import Dict, Tuple
import numpy as np


FEATURE_CHANNELS = (
    "gol_state",
    "membrane_v",
    "spike_map",
    "neighbor_count",
    "strategy_map",
    "memory_state",
)


def expected_shapes(height: int, width: int) -> Dict[str, Tuple[int, ...]]:
    """Expected tensor shapes per sample."""
    return {
        "X": (len(FEATURE_CHANNELS), height, width),
        "y": (),
    }


def validate_arrays(X: np.ndarray, y: np.ndarray) -> None:
    """Validate dataset arrays before writing to disk."""
    if X.ndim != 4:
        raise ValueError(f"X must be rank-4 [N,C,H,W], got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be rank-1 [N], got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y sample counts mismatch: {X.shape[0]} vs {y.shape[0]}")
