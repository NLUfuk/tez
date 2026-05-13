# Conway Game of Life + Izhikevich Neuron Hybrid Simulation

A hybrid simulation combining Conway's Game of Life (GoL) cellular automaton with Izhikevich spiking neuron dynamics. This project demonstrates bidirectional coupling between cellular automata and neural dynamics.

## Project Overview

This simulation implements:
- **Conway Game of Life**: Classic B3/S23 cellular automaton rules
- **Izhikevich Neurons**: Simplified spiking neuron model with membrane potential dynamics
- **Bidirectional Coupling**: 
  - GoL → Neurons: Alive cells and neighbor counts drive input current
  - Neurons → GoL: Spikes can influence Conway grid (optional feedback)
- **Unified Multi-Topology Runtime (Stream Mode)**:
  - Random small-world graph and multiple SWC morphology graphs can run together
  - Selected topologies are merged into one sparse unified graph
  - Frontend can hot-swap selected topology sets at runtime

## New in This Version: Multi-Topology + SWC

The stream pipeline now supports topology selection from the UI:

- `small_world` (existing random topology)
- `legacy_cluster` (legacy-style dense core + sparse periphery graph)
- SWC files from `data/morphology/` (e.g. `granule_test`, `medium_spiniy_test`, `pyramidal_test`)

At runtime, selected topologies are loaded and merged by a topology manager:

- **Spatial offsetting** avoids overlap in 3D
- **Sparse graph merging** uses SciPy CSR/block-diagonal assembly
- **One simulation engine view**: merged graph runs as a single network for Conway+Izhikevich steps

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

### Live Stream + Three.js Visualizer

Run backend stream server:

```bash
python -m scripts.run_live --mode stream --port 8765 --steps 50000 --interval 40
```

Run frontend static server:

```bash
python -m http.server 5173 --directory web
```

Open:

- `http://127.0.0.1:5173/three_visualizer/index.html`

#### One-command full stack startup (Windows PowerShell)

If you want backend + frontend to run together every time:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_full_stack.ps1
```

Useful overrides:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_full_stack.ps1 `
  -BackendPort 8765 -FrontendPort 5173 `
  -Topology "small_world,granule_test,pyramidal_test" `
  -SmallWorldN 2000 -Interval 40 -Steps 100000
```

The script keeps both services alive and stops both on `Ctrl+C`.
Logs are written under `outputs/live_logs/`.

#### Stream mode (new topology flags)

```bash
python -m scripts.run_live --mode stream \
  --topology small_world,granule_test,pyramidal_test \
  --swc-dir data/morphology \
  --small-world-n 3600
```

Key flags:

- `--topology`: initial comma-separated topology selection
- `--swc-dir`: directory containing `.swc` files
- `--small-world-n`: node count for the small-world component

HTTP endpoints:

- `GET /state`: latest frame payload
- `GET /topologies`: available topology options (small_world + discovered SWC files)
- `POST /control`: runtime controls (`set_topology`, coupling, rules, speed, reset, interactions)

## Topology & SWC Architecture

New backend modules:

- `src/conway_izh/swc_loader.py`:
  - Parses SWC rows `(id, type, x, y, z, radius, parent_id)`
  - Produces `coords (N,3)` + symmetric sparse adjacency (`scipy.sparse.csr_matrix`)
- `src/conway_izh/topology_manager.py`:
  - Discovers available SWC files
  - Builds selected components
  - Applies spatial offsets
  - Merges sparse adjacencies into unified graph
- `src/conway_izh/graph_grid.py`:
  - Graph-native simulation loop over 1D node state
  - Sparse neighbor counting and graph-based coupling

Updated stream publisher (`src/conway_izh/viz.py`) now supports:

- topology-aware payloads (`topology.active`, `selection`, `components`, `node_coords`)
- runtime `set_topology` hot swap
- `/topologies` endpoint

## Memory and Performance Notes (8 GB RAM / GTX 1650)

- Python graph operations use `scipy.sparse` matrices; no dense NxN adjacency allocation
- Frontend geometry attributes use typed arrays (`Float32Array`) only
- On topology rebuild, previous Three.js resources are explicitly disposed:
  - `BufferGeometry.dispose()`
  - `Material.dispose()`
  - old `Points`/`LineSegments` removed from scene
- Camera auto-fit uses scene bounds to frame all merged topologies

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
      swc_loader.py      # SWC parser -> coords + sparse adjacency
      topology_manager.py# Multi-topology selection and sparse merge
      graph_grid.py      # Graph-native simulation engine
  scripts/
    run_grid.py          # Grid simulation script
    run_single.py        # Single neuron script
    run_live.py          # Live stream server (matplotlib/stream mode)
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
- **swc_loader.py**: SWC parsing and sparse morphology graph extraction
- **topology_manager.py**: Unified sparse topology construction from selected components
- **graph_grid.py**: Conway+Izhikevich stepping on sparse arbitrary graphs

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

