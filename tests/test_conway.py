"""Tests for Conway Game of Life implementation."""

import numpy as np
import pytest
from conway_izh.conway import (
    initialize_conway, update_conway, count_neighbors, add_pattern
)


def test_block_still_life():
    """Test that a block (still life) remains unchanged after 2 steps."""
    # Block pattern (2x2 square)
    block = np.array([
        [1, 1],
        [1, 1]
    ], dtype=np.uint8)
    
    # Create 4x4 grid with block in center
    state = np.zeros((4, 4), dtype=np.uint8)
    state[1:3, 1:3] = block
    
    # Run 2 steps
    state_after_1 = update_conway(state, wrap_around=False)
    state_after_2 = update_conway(state_after_1, wrap_around=False)
    
    # Block should remain unchanged
    assert np.array_equal(state[1:3, 1:3], state_after_2[1:3, 1:3]), \
        "Block pattern should remain unchanged"


def test_blinker_oscillator():
    """Test that a blinker (oscillator) rotates after 1 step."""
    # Blinker pattern (horizontal line of 3)
    blinker_h = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=np.uint8)
    
    # Create 5x5 grid with blinker in center
    state = np.zeros((5, 5), dtype=np.uint8)
    state[1:4, 1:4] = blinker_h
    
    # Run 1 step
    state_after = update_conway(state, wrap_around=False)
    
    # Blinker should become vertical
    # Check center column
    center_col = state_after[1:4, 2]
    assert np.sum(center_col) == 3, "Blinker should become vertical"
    assert center_col[0] == 1 and center_col[1] == 1 and center_col[2] == 1


def test_neighbor_count():
    """Test neighbor counting."""
    # Simple pattern
    state = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    
    neighbors = count_neighbors(state, wrap_around=False)
    
    # Center cell should have 4 neighbors
    assert neighbors[1, 1] == 4
    
    # Corner cells should have fewer neighbors
    assert neighbors[0, 0] == 3  # Top-left corner (has 3 neighbors: right, down, down-right)


def test_initialization():
    """Test grid initialization."""
    state = initialize_conway(10, 10, seed=42, initial_density=0.3)
    
    assert state.shape == (10, 10)
    assert state.dtype == np.uint8
    assert np.all((state == 0) | (state == 1))


def test_add_pattern():
    """Test adding a pattern to grid."""
    state = np.zeros((5, 5), dtype=np.uint8)
    pattern = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    
    add_pattern(state, pattern, 1, 1)
    
    assert state[1, 1] == 1
    assert state[1, 2] == 1
    assert state[2, 1] == 1
    assert state[2, 2] == 1

