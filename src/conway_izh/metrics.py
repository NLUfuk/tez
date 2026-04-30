"""Metrics computation for Conway-Izhikevich simulation."""

import numpy as np
from typing import Dict, List
from conway_izh.efficiency import (
    compute_stability_score,
    compute_information_score,
    compute_memory_score,
    compute_cost_score,
    compute_efficiency_score,
)


def compute_metrics(
    gol_state: np.ndarray,
    spikes: np.ndarray,
    v: np.ndarray,
    step: int,
    *,
    mean_memory: float = 0.0,
    cooperative_ratio: float = 0.0,
    strategy_shift: float = 0.0,
    efficiency_weights: Dict[str, float] | None = None
) -> Dict[str, float]:
    """
    Compute metrics for a single simulation step.
    
    Args:
        gol_state: Conway grid state (H, W)
        spikes: Spike mask (H, W) bool
        v: Membrane potential (H, W)
        step: Current simulation step
        
    Returns:
        Dictionary of metric values
    """
    H, W = gol_state.shape
    total_cells = H * W
    
    alive_count = int(np.sum(gol_state))
    spike_count = int(np.sum(spikes))
    mean_v = float(np.mean(v))
    alive_ratio = alive_count / total_cells if total_cells > 0 else 0.0
    firing_rate = spike_count / total_cells if total_cells > 0 else 0.0

    stability_score = compute_stability_score(alive_ratio, firing_rate)
    information_score = compute_information_score(
        firing_rate, cooperative_ratio, strategy_shift
    )
    memory_score = compute_memory_score(mean_memory, strategy_shift)
    cost_score = compute_cost_score(firing_rate, spike_count, total_cells)
    efficiency_score = compute_efficiency_score(
        stability_score,
        information_score,
        memory_score,
        cost_score,
        weights=efficiency_weights
    )
    
    return {
        'step': step,
        'alive_count': alive_count,
        'alive_ratio': alive_ratio,
        'spike_count': spike_count,
        'mean_v': mean_v,
        'firing_rate': firing_rate,
        'mean_memory': mean_memory,
        'cooperative_ratio': cooperative_ratio,
        'strategy_shift': strategy_shift,
        'stability_score': stability_score,
        'information_score': information_score,
        'memory_score': memory_score,
        'cost_score': cost_score,
        'efficiency_score': efficiency_score,
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    """
    Aggregate metrics across all steps.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with arrays for each metric
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    for key in metrics_list[0].keys():
        aggregated[key] = np.array([m[key] for m in metrics_list])
    
    return aggregated

