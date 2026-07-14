# Sensing-vs-Simulation

Numerical experiments for the trade-off between metrological sensitivity and simulation accuracy.

The code studies perturbed Hamiltonians

```text
H(delta) = H0 + sum_i delta_i H_i
```

and compares exact simulation error with the local prediction from the quantum Fisher information matrix (QFIM).

## File Structure

```text
src.py
Core math: Hamiltonians, exact evolution, product formula, QFI, simulation error.

experiment_utils.py
Shared experiment utilities: configs, initial states, data saving, line/grid/scaling runners.

run_line.py
Run one-dimensional perturbation scans.

run_grid2d.py
Run two-dimensional perturbation scans.

run_state_comparison.py
Run fixed-n comparisons between different initial states.

run_qfi_scaling.py
Run QFI scaling with system size n.

line_figures.ipynb
Plot one line-scan dataset.

grid2d_figures.ipynb
Plot one grid2d dataset.

state_comparison.ipynb
Plot fixed-n initial-state comparison datasets.

qfi_scaling.ipynb
Plot QFI scaling datasets.
```

There is intentionally no single `run.py` now. Each experiment has its own runner, so the parameters you usually change are near the top of the corresponding file.

## Installation

Use your `jax` environment:

```bash
conda activate jax
pip install -r requirements.txt
```

If JAX prints a CUDA version warning but the script finishes and writes data, the run completed.

## Hamiltonian Convention

`Nearest_Neighbour_1d` in `src.py` uses

```text
H = sum_i (hx X_i + hy Y_i + hz Z_i)
  + sum_i (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1})
```

So

```text
hx, hy, hz    single-site fields
Jx, Jy, Jz    nearest-neighbour couplings
```

For example, the default TFIM-like base Hamiltonian is stored in `experiment_utils.py` as

```python
DEFAULT_H0 = {
    "name": "tfim",
    "params": {
        "Jx": 0.0,
        "Jy": 0.0,
        "Jz": 1.0,
        "hx": 0.5,
        "hy": 0.0,
        "hz": 0.0,
    },
}
```

which means

```text
H0 = 0.5 sum_i X_i + 1.0 sum_i Z_i Z_{i+1}
```

## Running Experiments

Run scripts from the project root:

```bash
conda activate jax
python run_line.py
python run_grid2d.py
python run_state_comparison.py
python run_qfi_scaling.py
```

Each script also has a tiny smoke-test mode:

```bash
python run_line.py --smoke
python run_grid2d.py --smoke
python run_state_comparison.py --smoke
python run_qfi_scaling.py --smoke
```

Smoke tests use small `n` and very few scan points. They are useful for checking that code changes still run.

## Where Data Goes

Generated data is grouped by experiment type:

```text
data/
  line/
    dataset_name/
      config.json
      results.npz

  grid2d/
    dataset_name/
      config.json
      results.npz

  state_comparison/
    dataset_name/
      config.json
      results.npz

  qfi_scaling/
    dataset_name/
      config.json
      results.npz
```

The folder name is generated automatically from the run parameters:

```text
ising1d__{H0 name}__{perturbation names}__{initial state or task}__{scan type}__N...
```

For `qfi_scaling`, the task already names the scan, so the form is:

```text
ising1d__{H0 name}__{perturbation names}__qfi-scaling__N...
```

`config.json` contains the full details needed to understand or reproduce the run. The folder name is only the quick human-readable summary.

Example dataset names:

```text
data/line/ising1d__tfim__xfield-zfield__plus__line__N10/
data/grid2d/ising1d__tfim__xfield-zfield__plus__grid2d__N10/
data/state_comparison/ising1d__tfim__xfield-zfield__ghz__line__N10/
data/qfi_scaling/ising1d__tfim__xfield-zfield__qfi-scaling__N2-N10/
```

Old datasets may still exist directly under `data/` from earlier versions. New scripts write into the grouped folders above.

## Changing Parameters

Change the parameters at the top of the relevant run script.

### Line Scan

Edit `run_line.py`:

```python
N = 10
TIME = 1.0
INITIAL_STATE = PLUS_STATE

SCAN = {
    "type": "line",
    "direction": [1.0, 0.0],
    "epsilon": {"start": -0.3, "stop": 0.3, "num": 10},
}
```

Here

```text
direction = [1.0, 0.0]    scan delta_0 only
direction = [0.0, 1.0]    scan delta_1 only
direction = [1.0, 1.0]    scan delta_0 + delta_1
```

The actual perturbation vector is

```text
delta = epsilon * direction
```

### 2D Grid

Edit `run_grid2d.py`:

```python
N = 10
TIME = 1.0

SCAN = {
    "type": "grid2d",
    "delta_0": {"start": -0.4, "stop": 0.4, "num": 41},
    "delta_1": {"start": -0.4, "stop": 0.4, "num": 41},
}
```

The cost scales like `num0 * num1`, so increasing both axes gets expensive quickly.

### State Comparison

Edit `run_state_comparison.py`:

```python
N = 10
TIME = 1.0

STATES = {
    "plus": {...},
    "GHZ": {...},
    "random product": {...},
    "chaotic evolved": {...},
}
```

This script runs one line scan per initial state and saves the results into `data/state_comparison/`.

### QFI Scaling

Edit `run_qfi_scaling.py`:

```python
SCAN = {
    "type": "qfi_scaling",
    "n_values": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "direction": [0.0, 1.0],
    "states": {
        "plus": PLUS_STATE,
        "GHZ": GHZ_STATE,
        "random product": RANDOM_PRODUCT_STATE,
        "chaotic evolved": CHAOTIC_EVOLVED_STATE,
    },
}
```

This computes

```text
direction.T @ QFIM @ direction
```

for each state and each system size.

## Changing H0 Or Perturbations

The shared defaults are in `experiment_utils.py`:

```python
DEFAULT_H0
DEFAULT_PERTURBATIONS
```

The default perturbations are

```text
delta_0: xfield = sum_i X_i
delta_1: zfield = sum_i Z_i
```

If you want a coupling perturbation, use the coupling parameters. For example,

```python
{
    "name": "zzcoupling",
    "delta_index": 0,
    "params": {
        "Jx": 0.0,
        "Jy": 0.0,
        "Jz": 1.0,
        "hx": 0.0,
        "hy": 0.0,
        "hz": 0.0,
    },
    "term": "sum_i Z_i Z_{i+1}",
}
```

Useful naming conventions:

```text
xfield        sum_i X_i
yfield        sum_i Y_i
zfield        sum_i Z_i
xxcoupling    sum_i X_i X_{i+1}
yycoupling    sum_i Y_i Y_{i+1}
zzcoupling    sum_i Z_i Z_{i+1}
```

## Initial States

Available built-in state configs are in `experiment_utils.py`:

```text
PLUS_STATE
GHZ_STATE
RANDOM_PRODUCT_STATE
CHAOTIC_EVOLVED_STATE
```

The state types supported by the builder are:

```text
plus
ghz
random_product
chaotic_evolved
computational_basis
```

For a random product state:

```python
{"type": "random_product", "seed": 7}
```

Each qubit is sampled independently on the Bloch sphere, then the full state is the tensor product.

## Results Format

### Line-Like Scans

Produced by `run_line.py` and `run_state_comparison.py`.

```text
epsilons         shape (num,)
direction        shape (num_perturbations,)
deltas           shape (num, num_perturbations)
exact_error      shape (num,)
qfi_error        shape (num,)
ratio            shape (num,)
qfi_directional  scalar
qfim             shape (num_perturbations, num_perturbations)
```

Example:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load(
    "data/line/ising1d__tfim__xfield-zfield__plus__line__N10/results.npz"
)

plt.plot(data["epsilons"], data["exact_error"], "o-", label="exact")
plt.plot(data["epsilons"], data["qfi_error"], "o-", label="QFI")
plt.legend()
plt.show()
```

### 2D Grid Scans

Produced by `run_grid2d.py`.

```text
delta_0_values    shape (num0,)
delta_1_values    shape (num1,)
exact_error_grid  shape (num1, num0)
qfi_error_grid    shape (num1, num0)
ratio_grid        shape (num1, num0)
qfim              shape (2, 2)
```

### QFI Scaling

Produced by `run_qfi_scaling.py`.

```text
n_values           shape (num_n,)
state_labels       shape (num_states,)
direction          shape (num_perturbations,)
qfi                shape (num_states, num_n)
scaling_exponents  shape (num_states,)
```

Example:

```python
data = np.load(
    "data/qfi_scaling/ising1d__tfim__zfield__qfi-scaling__N2-N10/results.npz"
)

for label, qfi in zip(data["state_labels"], data["qfi"]):
    plt.loglog(data["n_values"], qfi, "o-", label=str(label))

plt.legend()
plt.show()
```

## Plotting Notebooks

After generating data, open the matching notebook:

```text
line_figures.ipynb      reads data/line/
grid2d_figures.ipynb    reads data/grid2d/
state_comparison.ipynb  reads data/state_comparison/
qfi_scaling.ipynb       reads data/qfi_scaling/
```

The notebooks scan their data directory and choose a matching dataset automatically. They prefer full datasets over `__smoke` datasets.

## Product Formula

`src.py` contains a simple product-formula implementation:

```python
from src import product_formula

U = product_formula(H, time=1.0, reps=10, order=2)
```

Here

```text
time  total evolution time
reps  number of Trotter steps
dt    time / reps
order 1 for first-order Lie-Trotter
order 2 for second-order symmetric Suzuki-Trotter
```

Use a `SparsePauliOp` or a list of Hamiltonian terms if you want an actual product formula. If you pass a full dense matrix as one term, the function computes the exact exponential of that full matrix.
