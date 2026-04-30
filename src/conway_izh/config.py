"""Configuration dataclasses for Conway-Izhikevich simulation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IzhikevichParams:
    """Izhikevich neuron parameters."""
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0


@dataclass
class CouplingParams:
    """Coupling parameters between Conway and Izhikevich."""
    k_neighbors: float = 0.5  # Weight for neighbor count
    k_alive: float = 2.0      # Weight for alive cells
    bias: float = 0.0         # Constant bias
    feedback_enabled: bool = False  # Enable neuron->GoL feedback
    use_game_theory: bool = False  # Use game theory based spike generation
    propagation_strength: float = 0.5  # Spike propagation strength (0-1)
    cooperation_factor: float = 0.3  # How much propagation influences spike decision
    cooperation_strength: float = 0.7  # Feedback cooperation strength


@dataclass
class MemoryParams:
    """PDA-like lightweight memory dynamics per cell."""
    enabled: bool = True
    decay: float = 0.92
    spike_gain: float = 0.35
    neighbor_gain: float = 0.08
    clip_min: float = 0.0
    clip_max: float = 1.0


@dataclass
class StrategyParams:
    """Cell-wise strategy selection between greedy and cooperative behavior."""
    cellwise_enabled: bool = True
    greedy_bias: float = 0.08
    coop_bias: float = 0.08
    temperature: float = 6.0
    switch_cost: float = 0.03


@dataclass
class SimulationConfig:
    """Main simulation configuration."""
    # Grid dimensions
    height: int = 60
    width: int = 60
    
    # Simulation parameters
    steps: int = 300
    dt: float = 0.1  # Time step for neuron dynamics
    
    # Random seed
    seed: Optional[int] = 42
    
    # Conway parameters
    wrap_around: bool = False  # Boundary wrapping
    
    # Izhikevich parameters
    izh_params: IzhikevichParams = None
    
    # Coupling parameters
    coupling: CouplingParams = None
    memory: MemoryParams = None
    strategy: StrategyParams = None
    
    # Output settings
    output_dir: str = "outputs"
    run_id: Optional[str] = None
    save_gif: bool = False
    frame_stride: int = 5  # Save every N frames for GIF
    
    def __post_init__(self):
        """Initialize default nested dataclasses if None."""
        if self.izh_params is None:
            self.izh_params = IzhikevichParams()
        if self.coupling is None:
            self.coupling = CouplingParams()
        if self.memory is None:
            self.memory = MemoryParams()
        if self.strategy is None:
            self.strategy = StrategyParams()

