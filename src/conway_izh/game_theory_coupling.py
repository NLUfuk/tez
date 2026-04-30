"""Game theory and Conway-based spike coupling mechanisms."""

import numpy as np
from typing import Tuple
from conway_izh.config import CouplingParams, StrategyParams
from conway_izh.conway import count_neighbors


def conway_based_spike_trigger(
    gol_state: np.ndarray,
    neighbor_count: np.ndarray,
    v: np.ndarray,
    coupling: CouplingParams,
    wrap_around: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spikes based on Conway Game of Life principles.
    
    Game Theory Rules:
    1. Birth potential: Dead cells with exactly 3 neighbors have high spike probability
       (they want to "cooperate" to be born)
    2. Survival activity: Alive cells with 2-3 neighbors spike to maintain community
       (cooperation for survival)
    3. Overcrowding: Alive cells with >3 neighbors spike less (competition)
    4. Isolation: Alive cells with <2 neighbors spike less (loneliness)
    
    Args:
        gol_state: Binary Conway grid (H, W) dtype uint8
        neighbor_count: Neighbor count grid (H, W) dtype uint8
        v: Current membrane potential (H, W)
        coupling: Coupling parameters
        wrap_around: Boundary wrapping mode
        
    Returns:
        Tuple of (spike_probability, input_current)
        - spike_probability: Probability of spike (0-1)
        - input_current: Input current based on Conway state
    """
    H, W = gol_state.shape
    spike_prob = np.zeros((H, W), dtype=np.float64)
    I = np.zeros((H, W), dtype=np.float64)
    
    # Conway B3/S23 rules applied to spike generation
    dead_cells = (gol_state == 0)
    alive_cells = (gol_state == 1)
    
    # Rule 1: Birth potential (B3)
    # Dead cells with exactly 3 neighbors have high spike probability
    birth_condition = dead_cells & (neighbor_count == 3)
    spike_prob[birth_condition] = 0.8  # High probability to "cooperate" for birth
    I[birth_condition] = coupling.k_alive * 3.0 + coupling.bias
    
    # Rule 2: Survival activity (S23)
    # Alive cells with 2-3 neighbors spike to maintain community
    survival_condition = alive_cells & ((neighbor_count == 2) | (neighbor_count == 3))
    spike_prob[survival_condition] = 0.6  # Moderate probability for cooperation
    I[survival_condition] = coupling.k_alive * 2.5 + coupling.k_neighbors * neighbor_count[survival_condition] + coupling.bias
    
    # Rule 3: Overcrowding (competition)
    # Alive cells with >3 neighbors spike less (competition reduces cooperation)
    overcrowding = alive_cells & (neighbor_count > 3)
    spike_prob[overcrowding] = 0.2  # Low probability due to competition
    I[overcrowding] = coupling.k_alive * 1.0 + coupling.k_neighbors * neighbor_count[overcrowding] * 0.5 + coupling.bias
    
    # Rule 4: Isolation (loneliness)
    # Alive cells with <2 neighbors spike less
    isolation = alive_cells & (neighbor_count < 2)
    spike_prob[isolation] = 0.1  # Very low probability
    I[isolation] = coupling.k_alive * 0.5 + coupling.bias
    
    # Additional: Dead cells near birth threshold (2 neighbors) have moderate potential
    near_birth = dead_cells & (neighbor_count == 2)
    spike_prob[near_birth] = 0.3
    I[near_birth] = coupling.k_neighbors * 2.0 + coupling.bias
    
    return spike_prob, I


def propagate_spikes(
    spikes: np.ndarray,
    gol_state: np.ndarray,
    propagation_strength: float = 0.5,
    wrap_around: bool = False
) -> np.ndarray:
    """
    Propagate spikes to neighboring cells based on Conway connectivity.
    
    Game Theory: Spike propagation represents information sharing and cooperation.
    Cells that spike "communicate" with neighbors, increasing their spike probability.
    
    Args:
        spikes: Current spike mask (H, W) bool
        gol_state: Conway grid state (H, W)
        propagation_strength: Strength of spike propagation (0-1)
        wrap_around: Boundary wrapping mode
        
    Returns:
        Propagated spike influence mask (H, W) float64
    """
    if np.sum(spikes) == 0:
        return np.zeros_like(spikes, dtype=np.float64)
    
    H, W = spikes.shape
    propagation = np.zeros((H, W), dtype=np.float64)
    
    # For each spiking cell, propagate to neighbors
    spike_locations = np.where(spikes)
    
    for i, j in zip(spike_locations[0], spike_locations[1]):
        # Propagate to 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = i + di, j + dj
                
                if wrap_around:
                    ni = ni % H
                    nj = nj % W
                else:
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        continue
                
                # Propagation strength depends on Conway state
                # Alive neighbors receive stronger signal (cooperation)
                if gol_state[ni, nj] == 1:
                    propagation[ni, nj] += propagation_strength * 1.5
                else:
                    propagation[ni, nj] += propagation_strength * 0.8
    
    return propagation


def game_theory_spike_decision(
    spike_prob: np.ndarray,
    v: np.ndarray,
    propagation_influence: np.ndarray,
    v_threshold: float = 30.0,
    cooperation_factor: float = 0.3
) -> np.ndarray:
    """
    Make spike decision based on game theory and membrane potential.
    
    Decision combines:
    1. Conway-based spike probability
    2. Membrane potential threshold
    3. Propagation influence (cooperation signal)
    4. Random element (stochastic cooperation)
    
    Args:
        spike_prob: Conway-based spike probability (H, W)
        v: Membrane potential (H, W)
        propagation_influence: Spike propagation influence (H, W)
        v_threshold: Membrane potential threshold for spike
        cooperation_factor: How much propagation influences decision
        
    Returns:
        Final spike mask (H, W) bool
    """
    # Combine Conway probability with propagation influence
    combined_prob = spike_prob + cooperation_factor * propagation_influence
    combined_prob = np.clip(combined_prob, 0.0, 1.0)
    
    # Stochastic spike decision based on probability
    # Higher probability = more likely to spike
    random_values = np.random.random(spike_prob.shape)
    probability_spike = random_values < combined_prob
    
    # Also check membrane potential threshold
    # Cells near threshold are more likely to spike
    v_normalized = np.clip((v + 65) / 95.0, 0.0, 1.0)  # Normalize to 0-1
    v_based_spike = v_normalized > 0.7  # Spike if V is high enough
    
    # Final decision: spike if either condition is met
    final_spikes = probability_spike | v_based_spike
    
    return final_spikes.astype(bool)


def select_cellwise_strategies(
    gol_state: np.ndarray,
    neighbor_count: np.ndarray,
    propagation_influence: np.ndarray,
    memory_state: np.ndarray,
    previous_strategy_map: np.ndarray,
    strategy_params: StrategyParams
) -> np.ndarray:
    """
    Select cell-wise greedy/cooperative strategy map.

    Strategy encoding:
    - 0: greedy
    - 1: cooperative
    """
    if not strategy_params.cellwise_enabled:
        return np.ones_like(gol_state, dtype=np.int8)

    neighbor_norm = np.clip(neighbor_count / 8.0, 0.0, 1.0)
    propagation_norm = np.clip(propagation_influence, 0.0, 1.0)
    memory_norm = np.clip(memory_state, 0.0, 1.0)

    greedy_utility = (
        (1.0 - neighbor_norm)
        + strategy_params.greedy_bias
        + 0.25 * (1.0 - memory_norm)
    )
    cooperative_utility = (
        neighbor_norm
        + propagation_norm
        + 0.35 * memory_norm
        + strategy_params.coop_bias
    )

    # Penalize strategy flipping to stabilize dynamics.
    switch_to_greedy = previous_strategy_map == 1
    switch_to_coop = previous_strategy_map == 0
    greedy_utility = greedy_utility - strategy_params.switch_cost * switch_to_greedy
    cooperative_utility = cooperative_utility - strategy_params.switch_cost * switch_to_coop

    logits = strategy_params.temperature * (cooperative_utility - greedy_utility)
    coop_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    random_values = np.random.random(gol_state.shape)
    return (random_values < coop_prob).astype(np.int8)


def game_theory_spike_decision_with_strategy(
    spike_prob: np.ndarray,
    v: np.ndarray,
    propagation_influence: np.ndarray,
    strategy_map: np.ndarray,
    memory_state: np.ndarray,
    coupling: CouplingParams
) -> np.ndarray:
    """
    Strategy-aware spike decision.

    Cooperative cells rely more on propagation and memory.
    Greedy cells prioritize local spike potential and membrane readiness.
    """
    is_cooperative = strategy_map == 1
    is_greedy = ~is_cooperative

    memory_term = np.clip(memory_state, 0.0, 1.0)
    propagation_term = np.clip(propagation_influence, 0.0, 1.0)
    v_normalized = np.clip((v + 65) / 95.0, 0.0, 1.0)

    cooperative_prob = (
        spike_prob
        + coupling.cooperation_factor * propagation_term
        + 0.25 * memory_term
    )
    greedy_prob = (
        spike_prob
        + 0.35 * v_normalized
        + 0.12 * (1.0 - memory_term)
    )
    combined_prob = np.where(is_cooperative, cooperative_prob, greedy_prob)
    combined_prob = np.clip(combined_prob, 0.0, 1.0)

    probability_spike = np.random.random(spike_prob.shape) < combined_prob
    v_based_spike = v_normalized > 0.7
    return (probability_spike | v_based_spike).astype(bool)


def conway_to_current_game_theory(
    gol_state: np.ndarray,
    neighbor_count: np.ndarray,
    previous_spikes: np.ndarray,
    coupling: CouplingParams,
    wrap_around: bool = False,
    propagation_strength: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Conway state to input current using game theory principles.
    
    This function combines:
    1. Conway-based spike probability calculation
    2. Spike propagation from previous step
    3. Game theory decision making
    
    Args:
        gol_state: Binary Conway grid (H, W)
        neighbor_count: Neighbor count grid (H, W)
        previous_spikes: Spikes from previous step (H, W) bool
        coupling: Coupling parameters
        wrap_around: Boundary wrapping mode
        
    Returns:
        Tuple of (input_current, spike_probability)
    """
    # Calculate Conway-based spike probability and base current
    spike_prob, base_I = conway_based_spike_trigger(
        gol_state, neighbor_count, None, coupling, wrap_around
    )
    
    # Propagate spikes from previous step
    propagation = propagate_spikes(
        previous_spikes, gol_state, propagation_strength, wrap_around
    )
    
    # Add propagation influence to current
    I = base_I + coupling.k_neighbors * propagation * 2.0
    
    return I, spike_prob


def apply_game_theory_feedback(
    gol_state: np.ndarray,
    spikes: np.ndarray,
    neighbor_count: np.ndarray,
    cooperation_strength: float = 0.7
) -> np.ndarray:
    """
    Apply spike feedback to Conway grid using game theory.
    
    Rules:
    1. Spike locations become alive (cooperation)
    2. Spikes near dead cells with 2 neighbors help birth (cooperation)
    3. Multiple spikes in area create "cooperation zone"
    
    Args:
        gol_state: Current Conway grid (H, W)
        spikes: Spike mask (H, W) bool
        neighbor_count: Neighbor count (H, W)
        cooperation_strength: How strongly spikes affect Conway
        
    Returns:
        Modified Conway grid (H, W)
    """
    new_state = gol_state.copy()
    
    # Direct feedback: spikes make cells alive
    new_state[spikes] = 1
    
    # Cooperative birth: spikes near dead cells with 2 neighbors help birth
    dead_cells = (gol_state == 0)
    near_birth = dead_cells & (neighbor_count == 2)
    
    # Count spikes in neighborhood of near-birth cells
    H, W = gol_state.shape
    for i in range(H):
        for j in range(W):
            if near_birth[i, j]:
                # Check if any neighbor spiked
                spike_neighbors = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            if spikes[ni, nj]:
                                spike_neighbors += 1
                
                # If enough neighbors spiked, birth occurs (cooperation)
                if spike_neighbors >= 1 and np.random.random() < cooperation_strength:
                    new_state[i, j] = 1
    
    return new_state

