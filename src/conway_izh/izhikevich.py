"""Izhikevich neuron model implementation."""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class IzhikevichParams:
    """Izhikevich neuron parameters."""
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0


def step_izhikevich(v: np.ndarray, u: np.ndarray, I: np.ndarray,
                   params: IzhikevichParams, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single step of Izhikevich neuron model (vectorized).
    
    Equations:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        
    Spike condition: v >= 30 -> reset v = c, u += d
    
    Args:
        v: Membrane potential array (H, W) or (N,)
        u: Recovery variable array (H, W) or (N,)
        I: Input current array (H, W) or (N,)
        params: Izhikevich parameters
        dt: Time step
        
    Returns:
        Tuple of (v_new, u_new, spikes) where spikes is boolean array
    """
    # Update v and u
    dv = dt * (0.04 * v * v + 5 * v + 140 - u + I)
    du = dt * params.a * (params.b * v - u)
    
    v_new = v + dv
    u_new = u + du
    
    # Detect spikes
    spikes = v_new >= 30.0
    
    # Reset spiked neurons
    v_new[spikes] = params.c
    u_new[spikes] += params.d
    
    return v_new, u_new, spikes


def initialize_izhikevich(shape: Tuple[int, ...], seed: Optional[int] = None,
                         v_init: float = -65.0, u_init: float = -13.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize Izhikevich neuron states.
    
    Args:
        shape: Shape tuple (H, W) or (N,)
        seed: Random seed (for future use if needed)
        v_init: Initial membrane potential
        u_init: Initial recovery variable
        
    Returns:
        Tuple of (v, u) arrays
    """
    v = np.full(shape, v_init, dtype=np.float64)
    u = np.full(shape, u_init, dtype=np.float64)
    return v, u

