"""Efficiency score components for hybrid neural simulation."""

import numpy as np
from typing import Dict


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_stability_score(alive_ratio: float, firing_rate: float) -> float:
    """
    Favor mid-range activity to avoid collapse and chaos extremes.
    """
    alive_target = 0.35
    firing_target = 0.18
    alive_term = 1.0 - abs(alive_ratio - alive_target) / alive_target
    firing_term = 1.0 - abs(firing_rate - firing_target) / firing_target
    return _clip01(0.55 * alive_term + 0.45 * firing_term)


def compute_information_score(
    firing_rate: float,
    cooperative_ratio: float,
    strategy_shift: float
) -> float:
    """
    Reward meaningful communication with controlled adaptation.
    """
    firing_gain = np.tanh(2.5 * firing_rate)
    coop_gain = cooperative_ratio
    adaptation = 1.0 - np.clip(strategy_shift * 6.0, 0.0, 1.0)
    return _clip01(0.45 * firing_gain + 0.35 * coop_gain + 0.20 * adaptation)


def compute_memory_score(mean_memory: float, strategy_shift: float) -> float:
    """
    Favor persistent but not frozen memory traces.
    """
    memory_target = 0.45
    memory_quality = 1.0 - abs(mean_memory - memory_target) / memory_target
    adaptation_bonus = 1.0 - np.clip(strategy_shift * 4.0, 0.0, 1.0)
    return _clip01(0.75 * memory_quality + 0.25 * adaptation_bonus)


def compute_cost_score(firing_rate: float, spike_count: int, total_cells: int) -> float:
    """
    Higher score means higher cost (penalty term in efficiency).
    """
    if total_cells <= 0:
        return 0.0
    spike_density = spike_count / total_cells
    return _clip01(0.7 * spike_density + 0.3 * np.clip(firing_rate * 1.4, 0.0, 1.0))


def compute_efficiency_score(
    stability_score: float,
    information_score: float,
    memory_score: float,
    cost_score: float,
    weights: Dict[str, float] | None = None
) -> float:
    """
    Multi-objective efficiency score:
    E = w1*stability + w2*information + w3*memory - w4*cost
    """
    w = weights or {
        "stability": 0.30,
        "information": 0.30,
        "memory": 0.25,
        "cost": 0.15,
    }
    raw = (
        w["stability"] * stability_score
        + w["information"] * information_score
        + w["memory"] * memory_score
        - w["cost"] * cost_score
    )
    return _clip01(raw)
