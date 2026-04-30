# Conway Game of Life + Izhikevich Neuron Hybrid Simulation

A hybrid simulation combining Conway's Game of Life (GoL) cellular automaton with Izhikevich spiking neuron dynamics. This project demonstrates bidirectional coupling between cellular automata and neural dynamics.

## Project Overview

This simulation implements:
- **Conway Game of Life**: Classic B3/S23 cellular automaton rules
- **Izhikevich Neurons**: Simplified spiking neuron model with membrane potential dynamics
- **Bidirectional Coupling**: 
  - GoL → Neurons: Alive cells and neighbor counts drive input current
  - Neurons → GoL: Spikes can influence Conway grid (optional feedback)

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**Note**: Make sure to set `PYTHONPATH` to include the `src` directory, or install the package:
```bash
# Option 1: Set PYTHONPATH (Windows PowerShell)
$env:PYTHONPATH="C:\path\to\tez\src"

# Option 1: Set PYTHONPATH (Linux/Mac)
export PYTHONPATH="$(pwd)/src"

# Option 2: Install package in development mode
pip install -e .
```

### Grid Simulation

Run the main hybrid simulation:

```bash
python -m scripts.run_grid --height 60 --width 60 --steps 300 --seed 42 --gif --frame-stride 5
```

With feedback enabled:

```bash
python -m scripts.run_grid --height 60 --width 60 --steps 300 --seed 42 --feedback
```

**Command-line arguments:**
- `--height`, `--width`: Grid dimensions (default: 60x60)
- `--steps`: Number of simulation steps (default: 300)
- `--seed`: Random seed for reproducibility (default: None)
- `--out`: Output directory (default: outputs)
- `--run-id`: Custom run identifier (default: auto-generated timestamp)
- `--gif`: Generate GIF animation
- `--frame-stride`: Save every N frames for GIF (default: 5)
- `--wrap`: Enable wrap-around boundaries for Conway
- `--feedback`: Enable neuron→GoL feedback
- `--k-neighbors`: Coupling weight for neighbor count (default: 0.5)
- `--k-alive`: Coupling weight for alive cells (default: 2.0)
- `--bias`: Constant bias current (default: 0.0)
- `--izh-a`, `--izh-b`, `--izh-c`, `--izh-d`: Izhikevich parameters
- `--dt`: Time step for neuron dynamics (default: 0.1)

### Single Neuron Simulation

Run a single Izhikevich neuron:

```bash
python -m scripts.run_single --seed 1 --steps 1000 --I 10.0
```

## Output Files

Simulation outputs are saved to `outputs/<run_id>/`:

- **final_gol.png**: Final Conway Game of Life state
- **final_v.png**: Final membrane potential heatmap
- **spike_raster.png**: Spike raster plot (time × cell index)
- **metrics.csv**: Time series metrics (step, alive_count, spike_count, mean_v, firing_rate)
- **anim.gif**: Animation of Conway states (if `--gif` enabled)
- **frames/**: Individual frame images (if GIF enabled)

## Testing

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_conway.py
pytest tests/test_izhikevich.py
pytest tests/test_grid_smoke.py
```

## Architecture

### Package Structure

```
conway_izh/
  src/
    conway_izh/
      __init__.py
      config.py          # Configuration dataclasses
      conway.py          # Conway Game of Life logic
      izhikevich.py      # Izhikevich neuron model
      coupling.py        # GoL ↔ Neuron coupling
      grid.py            # NeuralGrid orchestrator
      viz.py             # Visualization functions
      metrics.py         # Metrics computation
  scripts/
    run_grid.py          # Grid simulation script
    run_single.py        # Single neuron script
  tests/
    test_conway.py       # Conway tests
    test_izhikevich.py   # Neuron tests
    test_grid_smoke.py   # Integration tests
```

### Module Responsibilities

- **config.py**: All simulation parameters as dataclasses
- **conway.py**: GoL state management, neighbor counting, B3/S23 update rule
- **izhikevich.py**: Vectorized neuron dynamics, spike detection
- **coupling.py**: Bidirectional signal conversion between systems
- **grid.py**: Main simulation loop orchestrator
- **viz.py**: PNG/GIF generation, CSV export
- **metrics.py**: Statistical metrics computation

## Parameters

### Izhikevich Parameters

- **a**: Recovery time constant (default: 0.02)
- **b**: Sensitivity of recovery variable (default: 0.2)
- **c**: Reset potential after spike (default: -65.0 mV)
- **d**: Recovery variable increment after spike (default: 8.0)

### Coupling Parameters

- **k_neighbors**: Weight for neighbor count contribution to current
- **k_alive**: Weight for alive cell contribution to current
- **bias**: Constant bias current
- **feedback_enabled**: Enable neuron spikes to influence GoL

### Conway Parameters

- **wrap_around**: Enable periodic boundaries (default: False)

## Performance

- **Time complexity**: O(H×W×T) per simulation
- **Space complexity**: O(H×W) for state storage
- Uses NumPy vectorization for efficient computation
- Deterministic with fixed seed

## Examples

### Basic Run
```bash
python -m scripts.run_grid --height 50 --width 50 --steps 200 --seed 42
```

### With Animation
```bash
python -m scripts.run_grid --height 100 --width 100 --steps 500 --gif --frame-stride 10
```

### Feedback Mode
```bash
python -m scripts.run_grid --height 60 --width 60 --steps 300 --feedback --k-alive 3.0
```

## License

This project is provided as-is for research and educational purposes.

