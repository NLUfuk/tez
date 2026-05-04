"""Conway Game of Life implementation using NumPy."""

import numpy as np
from typing import Sequence, Tuple


def count_neighbors(state: np.ndarray, wrap_around: bool = False) -> np.ndarray:
    """
    Count 8-neighbors for each cell in Conway grid.
    
    Args:
        state: Binary grid (H, W) with dtype uint8 (0/1)
        wrap_around: If True, wrap boundaries; if False, pad with zeros
        
    Returns:
        Neighbor count grid (H, W) with dtype uint8
    """
    H, W = state.shape
    
    if wrap_around:
        # Use np.roll for wrapping
        neighbors = np.zeros_like(state, dtype=np.uint8)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbors += np.roll(np.roll(state, di, axis=0), dj, axis=1)
    else:
        # Pad with zeros and use convolution-like approach
        padded = np.pad(state.astype(np.uint8), 1, mode='constant', constant_values=0)
        neighbors = np.zeros_like(state, dtype=np.uint8)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbors += padded[1+di:H+1+di, 1+dj:W+1+dj]
    
    return neighbors


def update_conway(
    state: np.ndarray,
    wrap_around: bool = False,
    birth_neighbors: Sequence[int] = (3,),
    survive_neighbors: Sequence[int] = (2, 3),
) -> np.ndarray:
    """
    Update Conway Game of Life state using configurable B/S rules.
    
    Args:
        state: Current binary grid (H, W) dtype uint8
        wrap_around: Boundary wrapping mode
        
    Returns:
        Updated binary grid (H, W) dtype uint8
    """
    neighbors = count_neighbors(state, wrap_around)
    
    birth_mask = np.isin(neighbors, np.asarray(tuple(birth_neighbors), dtype=np.uint8))
    survive_mask = np.isin(neighbors, np.asarray(tuple(survive_neighbors), dtype=np.uint8))
    birth = birth_mask & (state == 0)
    survive = survive_mask & (state == 1)
    
    new_state = np.zeros_like(state, dtype=np.uint8)
    new_state[birth | survive] = 1
    
    return new_state


def initialize_conway(height: int, width: int, seed: int = None, 
                     initial_density: float = 0.3) -> np.ndarray:
    """
    Initialize Conway grid with random pattern.
    
    Args:
        height: Grid height
        width: Grid width
        seed: Random seed
        initial_density: Probability of cell being alive
        
    Returns:
        Initial binary grid (H, W) dtype uint8
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    state = (rng.random((height, width)) < initial_density).astype(np.uint8)
    return state


def add_pattern(state: np.ndarray, pattern: np.ndarray, 
                row: int, col: int) -> np.ndarray:
    """
    Add a predefined pattern to the grid at specified position.
    
    Args:
        state: Grid to modify
        pattern: Binary pattern array (H_pattern, W_pattern)
        row: Top-left row position
        col: Top-left column position
        
    Returns:
        Modified grid (in-place modification)
    """
    H, W = state.shape
    H_pat, W_pat = pattern.shape
    
    # Clip to valid range
    r_end = min(row + H_pat, H)
    c_end = min(col + W_pat, W)
    r_start = max(0, row)
    c_start = max(0, col)
    
    pat_r_start = r_start - row
    pat_c_start = c_start - col
    pat_r_end = pat_r_start + (r_end - r_start)
    pat_c_end = pat_c_start + (c_end - c_start)
    
    state[r_start:r_end, c_start:c_end] = pattern[pat_r_start:pat_r_end, pat_c_start:pat_c_end]
    return state

