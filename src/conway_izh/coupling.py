"""Coupling mechanisms between Conway Game of Life and Izhikevich neurons."""

import numpy as np
from typing import Tuple
from conway_izh.config import CouplingParams


def gol_to_current(gol_state: np.ndarray, neighbor_count: np.ndarray,
                   coupling: CouplingParams) -> np.ndarray:
    """
    Convert Conway Game of Life state to input current for neurons.
    
    I = k_neighbors * neighbors + k_alive * alive + bias
    
    Args:
        gol_state: Binary Conway grid (H, W) dtype uint8
        neighbor_count: Neighbor count grid (H, W) dtype uint8
        coupling: Coupling parameters
        
    Returns:
        Input current array (H, W) dtype float64
    """
    I = (coupling.k_neighbors * neighbor_count.astype(np.float64) +
         coupling.k_alive * gol_state.astype(np.float64) +
         coupling.bias)
    return I


def neuron_to_gol_feedback(gol_state: np.ndarray, spikes: np.ndarray) -> np.ndarray:
    """
    Apply neuron spike feedback to Conway grid.
    
    Simple rule: spike locations become alive in next step.
    
    Args:
        gol_state: Current Conway grid (H, W) dtype uint8
        spikes: Spike mask (H, W) dtype bool
        
    Returns:
        Modified Conway grid (H, W) dtype uint8
    """
    # Where spikes occur, set alive
    new_state = gol_state.copy()
    new_state[spikes] = 1
    return new_state

