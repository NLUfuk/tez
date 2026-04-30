"""Tests for Izhikevich neuron implementation."""

import numpy as np
import pytest
from conway_izh.izhikevich import (
    initialize_izhikevich, step_izhikevich, IzhikevichParams
)


def test_izhikevich_spike():
    """Test that neuron produces spikes with sufficient input current."""
    params = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    
    # Initialize single neuron
    v, u = initialize_izhikevich((1,), seed=42)
    
    # Strong input current to ensure spikes
    I = np.array([20.0])  # Strong enough to cause spikes
    dt = 0.1
    
    spike_count = 0
    max_steps = 1000
    
    for step in range(max_steps):
        v_new, u_new, spikes = step_izhikevich(v, u, I, params, dt)
        v = v_new
        u = u_new
        
        if spikes[0]:
            spike_count += 1
            # After spike, v should be reset to c
            assert np.isclose(v[0], params.c, atol=1.0), \
                f"After spike, v should be reset to c={params.c}, got {v[0]}"
        
        # Check that v doesn't explode
        assert np.abs(v[0]) < 1000, "Membrane potential should not explode"
    
    assert spike_count > 0, "Neuron should produce at least one spike with I=20"


def test_izhikevich_no_spike():
    """Test that neuron doesn't spike with weak input."""
    params = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    
    v, u = initialize_izhikevich((1,), seed=42)
    
    # Weak input current
    I = np.array([0.0])
    dt = 0.1
    
    spike_count = 0
    steps = 100
    
    for step in range(steps):
        v_new, u_new, spikes = step_izhikevich(v, u, I, params, dt)
        v = v_new
        u = u_new
        
        if spikes[0]:
            spike_count += 1
    
    # With I=0, neuron should not spike (or spike very rarely)
    # This is a probabilistic test, but with I=0 it's very unlikely
    assert spike_count == 0 or spike_count <= 1, \
        f"With I=0, neuron should not spike frequently, got {spike_count} spikes"


def test_izhikevich_vectorized():
    """Test vectorized operation on multiple neurons."""
    params = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    
    # Initialize 2x2 grid of neurons
    v, u = initialize_izhikevich((2, 2), seed=42)
    
    # Different input currents
    I = np.array([[10.0, 5.0], [0.0, 15.0]])
    dt = 0.1
    
    # Run a few steps
    for step in range(50):
        v_new, u_new, spikes = step_izhikevich(v, u, I, params, dt)
        v = v_new
        u = u_new
    
    # Check shapes are preserved
    assert v.shape == (2, 2)
    assert u.shape == (2, 2)
    assert spikes.shape == (2, 2)


def test_initialization():
    """Test neuron initialization."""
    v, u = initialize_izhikevich((5, 5), seed=42)
    
    assert v.shape == (5, 5)
    assert u.shape == (5, 5)
    assert np.allclose(v, -65.0)  # Default v_init
    assert np.allclose(u, -13.0)  # Default u_init

