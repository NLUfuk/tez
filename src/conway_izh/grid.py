"""NeuralGrid orchestrator for Conway-Izhikevich hybrid simulation."""

import numpy as np
from pathlib import Path
from typing import List, Optional
import time

from conway_izh.config import SimulationConfig
from conway_izh.conway import (
    initialize_conway, update_conway, count_neighbors
)
from conway_izh.izhikevich import (
    initialize_izhikevich, step_izhikevich, IzhikevichParams
)
from conway_izh.coupling import gol_to_current, neuron_to_gol_feedback
from conway_izh.game_theory_coupling import (
    conway_to_current_game_theory,
    game_theory_spike_decision,
    apply_game_theory_feedback,
    select_cellwise_strategies,
    game_theory_spike_decision_with_strategy
)
from conway_izh.metrics import compute_metrics
from conway_izh.viz import (
    save_gol_frame, save_v_heatmap, save_spike_raster,
    save_metrics_csv, create_gif
)


class NeuralGrid:
    """
    Main orchestrator for Conway Game of Life + Izhikevich neuron simulation.
    
    This class manages the hybrid simulation loop, coupling between systems,
    and output generation.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize NeuralGrid with configuration.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        
        # Initialize Conway grid
        self.gol_state = initialize_conway(
            config.height, config.width, config.seed
        )
        
        # Initialize Izhikevich neurons
        self.v, self.u = initialize_izhikevich(
            (config.height, config.width), config.seed
        )
        
        # History for visualization
        self.spike_history: List[np.ndarray] = []
        self.metrics_history: List[dict] = []
        self.frame_paths: List[Path] = []
        
        # Previous spikes for game theory propagation
        self.previous_spikes: np.ndarray = np.zeros(
            (config.height, config.width), dtype=bool
        )
        self.memory_state: np.ndarray = np.zeros(
            (config.height, config.width), dtype=np.float64
        )
        self.strategy_map: np.ndarray = np.ones(
            (config.height, config.width), dtype=np.int8
        )
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        if config.run_id:
            self.output_dir = self.output_dir / config.run_id
        else:
            timestamp = int(time.time())
            self.output_dir = self.output_dir / f"run_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def step(self) -> tuple[dict, np.ndarray]:
        """
        Perform one simulation step.
        
        Strategy C: GoL updates separately, neurons update separately,
        coupling happens between updates.
        
        If game_theory mode is enabled:
        - Spikes are generated based on Conway rules (B3/S23)
        - Spike propagation creates connections between cells
        - Game theory decision making determines spike probability
        
        Returns:
            Tuple of (metrics dictionary, spikes array)
        """
        # Count neighbors for coupling
        neighbor_count = count_neighbors(
            self.gol_state, self.config.wrap_around
        )
        
        if self.config.coupling.use_game_theory:
            # Game theory based coupling
            # Calculate input current and spike probability from Conway state
            I, spike_prob = conway_to_current_game_theory(
                self.gol_state,
                neighbor_count,
                self.previous_spikes,
                self.config.coupling,
                self.config.wrap_around,
                self.config.coupling.propagation_strength
            )
            
            # Update neurons with calculated current
            self.v, self.u, neuron_spikes = step_izhikevich(
                self.v, self.u, I, self.config.izh_params, self.config.dt
            )
            
            # Game theory spike decision combines:
            # 1. Conway-based probability
            # 2. Neuron membrane potential
            # 3. Propagation influence
            from conway_izh.game_theory_coupling import propagate_spikes
            propagation = propagate_spikes(
                self.previous_spikes,
                self.gol_state,
                self.config.coupling.propagation_strength,
                self.config.wrap_around
            )

            # Cell-wise strategy selection (0=greedy, 1=cooperative)
            self.strategy_map = select_cellwise_strategies(
                self.gol_state,
                neighbor_count,
                propagation,
                self.memory_state,
                self.strategy_map,
                self.config.strategy
            )
            
            # Make final spike decision
            if self.config.strategy.cellwise_enabled:
                spikes = game_theory_spike_decision_with_strategy(
                    spike_prob,
                    self.v,
                    propagation,
                    self.strategy_map,
                    self.memory_state,
                    self.config.coupling
                )
            else:
                spikes = game_theory_spike_decision(
                    spike_prob,
                    self.v,
                    propagation,
                    v_threshold=30.0,
                    cooperation_factor=self.config.coupling.cooperation_factor
                )
            
            # Also include neuron spikes (if membrane potential threshold reached)
            spikes = spikes | neuron_spikes
            
        else:
            # Original coupling method
            # Convert GoL state to input current
            I = gol_to_current(
                self.gol_state, neighbor_count, self.config.coupling
            )
            
            # Update neurons
            self.v, self.u, spikes = step_izhikevich(
                self.v, self.u, I, self.config.izh_params, self.config.dt
            )
        
        # Update Conway grid
        new_gol_state = update_conway(
            self.gol_state, self.config.wrap_around
        )
        
        # Apply neuron feedback
        if self.config.coupling.feedback_enabled:
            if self.config.coupling.use_game_theory:
                # Game theory feedback
                from conway_izh.game_theory_coupling import apply_game_theory_feedback
                new_gol_state = apply_game_theory_feedback(
                    new_gol_state,
                    spikes,
                    neighbor_count,
                    self.config.coupling.cooperation_strength
                )
            else:
                # Original feedback
                new_gol_state = neuron_to_gol_feedback(new_gol_state, spikes)
        
        self.gol_state = new_gol_state
        
        if self.config.memory.enabled:
            spike_term = spikes.astype(np.float64) * self.config.memory.spike_gain
            neighbor_term = (
                neighbor_count.astype(np.float64) / 8.0
            ) * self.config.memory.neighbor_gain
            self.memory_state = (
                self.config.memory.decay * self.memory_state
                + spike_term
                + neighbor_term
            )
            self.memory_state = np.clip(
                self.memory_state,
                self.config.memory.clip_min,
                self.config.memory.clip_max
            )

        # Store spikes for next step propagation
        self.previous_spikes = spikes.copy()
        
        # Compute metrics
        mean_memory = float(np.mean(self.memory_state))
        cooperative_ratio = float(np.mean(self.strategy_map == 1))
        if self.metrics_history:
            prev_ratio = self.metrics_history[-1].get("cooperative_ratio", cooperative_ratio)
            strategy_shift = abs(cooperative_ratio - prev_ratio)
        else:
            strategy_shift = 0.0
        metrics = compute_metrics(
            self.gol_state,
            spikes,
            self.v,
            len(self.metrics_history),
            mean_memory=mean_memory,
            cooperative_ratio=cooperative_ratio,
            strategy_shift=strategy_shift,
        )
        
        return metrics, spikes
    
    def run(self):
        """Run full simulation."""
        print(f"Starting simulation: {self.config.steps} steps")
        print(f"Grid size: {self.config.height}x{self.config.width}")
        print(f"Output directory: {self.output_dir}")
        
        frames_dir = self.output_dir / "frames"
        if self.config.save_gif:
            frames_dir.mkdir(exist_ok=True)
        
        for step in range(self.config.steps):
            metrics, spikes = self.step()
            self.metrics_history.append(metrics)
            self.spike_history.append(spikes.copy())
            
            # Save frames for GIF if enabled
            if self.config.save_gif and step % self.config.frame_stride == 0:
                frame_path = frames_dir / f"frame_{step:06d}.png"
                save_gol_frame(self.gol_state, frame_path)
                self.frame_paths.append(frame_path)
            
            if step % 50 == 0:
                print(f"Step {step}/{self.config.steps}: "
                      f"alive={metrics['alive_count']}, "
                      f"spikes={metrics['spike_count']}, "
                      f"E={metrics['efficiency_score']:.3f}")
        
        print("Simulation complete. Generating outputs...")
        self._generate_outputs()
    
    def _generate_outputs(self):
        """Generate all output files."""
        # Final GoL state
        save_gol_frame(
            self.gol_state,
            self.output_dir / "final_gol.png"
        )
        
        # Final membrane potential
        save_v_heatmap(
            self.v,
            self.output_dir / "final_v.png"
        )
        
        # Metrics CSV
        save_metrics_csv(
            self.metrics_history,
            self.output_dir / "metrics.csv"
        )
        
        # Spike raster plot
        if self.spike_history:
            save_spike_raster(
                self.spike_history,
                self.output_dir / "spike_raster.png"
            )
        
        # Create GIF if enabled and frames exist
        if self.config.save_gif and self.frame_paths:
            create_gif(
                self.frame_paths,
                self.output_dir / "anim.gif",
                duration=0.1
            )
        
        print(f"Outputs saved to: {self.output_dir}")

