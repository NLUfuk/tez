"""Smoke tests for NeuralGrid."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conway_izh.grid import NeuralGrid
from conway_izh.config import SimulationConfig, IzhikevichParams, CouplingParams


def test_grid_smoke():
    """Smoke test: small grid should run without errors."""
    config = SimulationConfig(
        height=10,
        width=10,
        steps=5,
        dt=0.1,
        seed=42,
        output_dir="outputs/test_smoke",
        run_id="smoke_test",
        save_gif=False
    )
    
    grid = NeuralGrid(config)
    grid.run()
    
    # Check that output directory exists
    assert grid.output_dir.exists()
    
    # Check that metrics.csv was created
    metrics_file = grid.output_dir / "metrics.csv"
    assert metrics_file.exists(), "metrics.csv should be created"
    
    # Check that metrics.csv has correct number of rows (header + steps)
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == config.steps + 1, \
            f"metrics.csv should have {config.steps + 1} lines (header + {config.steps} steps)"
    
    # Check that final images were created
    assert (grid.output_dir / "final_gol.png").exists()
    assert (grid.output_dir / "final_v.png").exists()
    assert (grid.output_dir / "spike_raster.png").exists()


def test_grid_with_feedback():
    """Test grid with feedback enabled."""
    config = SimulationConfig(
        height=10,
        width=10,
        steps=5,
        dt=0.1,
        seed=42,
        coupling=CouplingParams(feedback_enabled=True),
        output_dir="outputs/test_feedback",
        run_id="feedback_test"
    )
    
    grid = NeuralGrid(config)
    grid.run()
    
    assert grid.output_dir.exists()
    assert (grid.output_dir / "metrics.csv").exists()


def test_grid_deterministic():
    """Test that same seed produces same initial state."""
    config1 = SimulationConfig(
        height=10,
        width=10,
        steps=1,
        seed=42
    )
    
    config2 = SimulationConfig(
        height=10,
        width=10,
        steps=1,
        seed=42
    )
    
    grid1 = NeuralGrid(config1)
    grid2 = NeuralGrid(config2)
    
    # Initial states should be identical
    assert np.array_equal(grid1.gol_state, grid2.gol_state), \
        "Same seed should produce same initial Conway state"
    assert np.allclose(grid1.v, grid2.v), \
        "Same seed should produce same initial neuron states"

