import copy
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import expm_multiply

from src import General_QFI, General_Simulation_error, Nearest_Neighbour_1d, expH


DATA_DIR = Path("data")

HAMILTONIAN_FORMULA = (
    "H = sum_i (hx X_i + hy Y_i + hz Z_i) "
    "+ sum_i (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1})"
)

DEFAULT_H0 = {
    "name": "tfim",
    "description": "Transverse-field Ising base Hamiltonian",
    "params": {
        "Jx": 0.0,
        "Jy": 0.0,
        "Jz": 1.0,
        "hx": 0.5,
        "hy": 0.0,
        "hz": 0.0,
    },
    "term": "0.5 sum_i X_i + 1.0 sum_i Z_i Z_{i+1}",
}

DEFAULT_PERTURBATIONS = [
    {
        "name": "xfield",
        "delta_index": 0,
        "description": "Global X-field calibration error",
        "params": {
            "Jx": 0.0,
            "Jy": 0.0,
            "Jz": 0.0,
            "hx": 1.0,
            "hy": 0.0,
            "hz": 0.0,
        },
        "term": "sum_i X_i",
    },
    {
        "name": "zfield",
        "delta_index": 1,
        "description": "Unmodeled global Z-field drift",
        "params": {
            "Jx": 0.0,
            "Jy": 0.0,
            "Jz": 0.0,
            "hx": 0.0,
            "hy": 0.0,
            "hz": 1.0,
        },
        "term": "sum_i Z_i",
    },
]

PLUS_STATE = {
    "type": "plus",
    "description": "Product state |+>^n",
}

GHZ_STATE = {
    "type": "ghz",
    "description": "GHZ state (|0...0> + |1...1>) / sqrt(2)",
}

RANDOM_PRODUCT_STATE = {
    "type": "random_product",
    "description": "Random product state sampled from Bloch sphere angles",
    "seed": 7,
}

CHAOTIC_EVOLVED_STATE = {
    "type": "chaotic_evolved",
    "description": "Thermal-like pure state from tilted-Ising evolution of |+>^n",
    "base_state": PLUS_STATE,
    "time": 2.5,
    "hamiltonian": {
        "name": "tilted_ising",
        "params": {
            "Jx": 0.0,
            "Jy": 0.0,
            "Jz": 1.0,
            "hx": 0.9,
            "hy": 0.0,
            "hz": 0.7,
        },
        "term": "0.9 sum_i X_i + 0.7 sum_i Z_i + 1.0 sum_i Z_i Z_{i+1}",
    },
}


def make_config(
    *,
    experiment_name,
    data_subdir,
    n,
    time,
    initial_state,
    scan,
    dataset_base=None,
    smoke=False,
    h0=None,
    perturbations=None,
):
    h0_config = copy.deepcopy(h0 or DEFAULT_H0)
    perturbation_configs = copy.deepcopy(perturbations or DEFAULT_PERTURBATIONS)
    initial_state_config = copy.deepcopy(initial_state)
    scan_config = copy.deepcopy(scan)
    if dataset_base is None:
        dataset_base = automatic_dataset_base(
            h0_config,
            perturbation_configs,
            initial_state_config,
            scan_config,
        )
    if smoke:
        dataset_base = f"{dataset_base}__smoke"

    config = {
        "experiment_name": experiment_name,
        "model": "Nearest_Neighbour_1d",
        "hamiltonian_formula": HAMILTONIAN_FORMULA,
        "n": n,
        "time": time,
        "initial_state": initial_state_config,
        "H0": h0_config,
        "perturbations": perturbation_configs,
        "scan": scan_config,
        "dataset_base": dataset_base,
        "data_subdir": data_subdir,
        "results_file": "results.npz",
    }
    config["dataset"] = make_dataset_name(config)
    config["delta_indices"] = {
        f"delta_{item['delta_index']}": item["name"]
        for item in config["perturbations"]
    }
    return config


def automatic_dataset_base(h0, perturbations, initial_state, scan):
    if scan["type"] == "qfi_scaling":
        return "__".join(
            [
                "ising1d",
                slug(h0.get("name", "h0")),
                "-".join(
                    slug(item.get("name", f"delta{index}"))
                    for index, item in enumerate(perturbations)
                ),
                "qfi-scaling",
            ]
        )
    if scan["type"] == "trotter_error":
        return "__".join(
            [
                "ising1d",
                slug(h0.get("name", "h0")),
                "trotter-error",
            ]
        )

    parts = [
        "ising1d",
        slug(h0.get("name", "h0")),
        "-".join(slug(item.get("name", f"delta{index}")) for index, item in enumerate(perturbations)),
        state_slug(initial_state, scan),
        scan_slug(scan),
    ]
    return "__".join(parts)


def state_slug(initial_state, scan):
    if scan["type"] == "qfi_scaling":
        return "qfi-scaling"
    state_type = initial_state["type"]
    if state_type == "random_product":
        return "random-product"
    if state_type == "chaotic_evolved":
        return "chaotic-evolved"
    return slug(state_type)


def scan_slug(scan):
    if scan["type"] == "qfi_scaling":
        return "qfi-scaling"
    return slug(scan["type"])


def slug(value):
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def linspace_from_config(config):
    return np.linspace(config["start"], config["stop"], config["num"])


def make_dataset_name(config):
    if config["scan"]["type"] == "qfi_scaling":
        values = config["scan"]["n_values"]
        return f"{config['dataset_base']}__N{values[0]}-N{values[-1]}"
    return f"{config['dataset_base']}__N{config['n']}"


def prepare_output_dir(config):
    output_dir = DATA_DIR / config["data_subdir"] / config["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config(output_dir, config)
    return output_dir


def write_config(output_dir, config):
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def build_problem(config):
    n = config["n"]
    psi0 = make_initial_state(config["initial_state"], n)
    h0 = hamiltonian_matrix(n, config["H0"]["params"])
    perturbations = [
        hamiltonian_matrix(n, item["params"])
        for item in config["perturbations"]
    ]
    return psi0, jnp.stack([h0, *perturbations])


def hamiltonian_matrix(n, params):
    return jnp.asarray(Nearest_Neighbour_1d(n=n, **params).ham.to_matrix())


def build_sparse_h_list(config):
    n = config["n"]
    h0 = hamiltonian_sparse_matrix(n, config["H0"]["params"])
    perturbations = [
        hamiltonian_sparse_matrix(n, item["params"])
        for item in config["perturbations"]
    ]
    return [h0, *perturbations]


def hamiltonian_sparse_matrix(n, params):
    return Nearest_Neighbour_1d(n=n, **params).ham.to_matrix(sparse=True)


def run_line_scan(config):
    psi0, h_list = build_problem(config)
    psi0_np = np.asarray(psi0)
    h_list_np = np.asarray(h_list)
    time = config["time"]
    direction = jnp.asarray(config["scan"]["direction"], dtype=float)
    direction_np = np.asarray(direction)
    epsilons = linspace_from_config(config["scan"]["epsilon"])
    deltas = np.asarray([epsilon * direction_np for epsilon in epsilons])
    ideal_state = evolve_state(h_list_np[0], psi0_np, time)

    qfim = np.asarray(General_QFI(jnp.zeros(len(direction_np)), psi0, h_list, time))
    qfi_directional = directional_qfi(direction_np, qfim)
    exact_error = np.asarray(
        [
            simulation_error_np(delta, psi0_np, h_list_np, time, ideal_state)
            for delta in deltas
        ]
    )
    qfi_error = 0.5 * np.abs(epsilons) * np.sqrt(max(qfi_directional, 0.0))
    ratio = safe_ratio(exact_error**2, qfi_error**2)

    output_dir = prepare_output_dir(config)
    np.savez(
        output_dir / config["results_file"],
        epsilons=epsilons,
        direction=np.asarray(direction),
        deltas=deltas,
        exact_error=exact_error,
        qfi_error=qfi_error,
        ratio=ratio,
        qfi_directional=qfi_directional,
        qfim=qfim,
    )
    return output_dir


def run_grid2d_scan(config):
    psi0, h_list = build_problem(config)
    psi0_np = np.asarray(psi0)
    h_list_np = build_sparse_h_list(config)
    time = config["time"]
    delta_0_values = linspace_from_config(config["scan"]["delta_0"])
    delta_1_values = linspace_from_config(config["scan"]["delta_1"])
    ideal_state = evolve_state(h_list_np[0], psi0_np, time)

    qfim = np.asarray(General_QFI(jnp.zeros(2), psi0, h_list, time))
    exact_error_grid = np.zeros((len(delta_1_values), len(delta_0_values)))
    qfi_error_grid = np.zeros_like(exact_error_grid)

    for row, delta_1 in enumerate(delta_1_values):
        for col, delta_0 in enumerate(delta_0_values):
            delta = np.asarray([delta_0, delta_1])
            exact_error_grid[row, col] = simulation_error_np(
                delta, psi0_np, h_list_np, time, ideal_state
            )
            qfi_error_grid[row, col] = quadratic_error(delta[None, :], qfim)[0]

    ratio_grid = safe_ratio(exact_error_grid**2, qfi_error_grid**2)
    output_dir = prepare_output_dir(config)
    np.savez(
        output_dir / config["results_file"],
        delta_0_values=delta_0_values,
        delta_1_values=delta_1_values,
        exact_error_grid=exact_error_grid,
        qfi_error_grid=qfi_error_grid,
        ratio_grid=ratio_grid,
        qfim=qfim,
    )
    return output_dir


def run_qfi_scaling(config):
    n_values = np.asarray(config["scan"]["n_values"], dtype=int)
    direction = np.asarray(config["scan"]["direction"], dtype=float)
    state_items = list(config["scan"]["states"].items())
    state_labels = np.asarray([label for label, _ in state_items])
    qfi_values = np.zeros((len(state_items), len(n_values)))

    for state_index, (_, state_config) in enumerate(state_items):
        for n_index, n in enumerate(n_values):
            local_config = copy.deepcopy(config)
            local_config["n"] = int(n)
            local_config["initial_state"] = copy.deepcopy(state_config)
            psi0, h_list = build_problem(local_config)
            qfim = np.asarray(
                General_QFI(jnp.zeros(len(direction)), psi0, h_list, local_config["time"])
            )
            qfi_values[state_index, n_index] = directional_qfi(direction, qfim)

    scaling_exponents = np.asarray(
        [fit_power_law(n_values, values) for values in qfi_values]
    )

    output_dir = prepare_output_dir(config)
    np.savez(
        output_dir / config["results_file"],
        n_values=n_values,
        state_labels=state_labels,
        direction=direction,
        qfi=qfi_values,
        scaling_exponents=scaling_exponents,
    )
    return output_dir


def directional_qfi(direction, qfim):
    return float(np.einsum("i,ij,j->", direction, qfim, direction))


def simulation_error_np(delta, psi0, h_list, time, ideal_state):
    hamiltonian = construct_h_np(delta, h_list)
    real_state = evolve_state(hamiltonian, psi0, time)
    overlap = np.vdot(ideal_state, real_state)
    return pure_state_bures_distance_np(overlap)


def pure_state_bures_distance_np(overlap):
    fidelity_sqrt = min(abs(overlap), 1.0)
    return np.sqrt(max(0.0, 2.0 * (1.0 - fidelity_sqrt)))


def evolve_state(hamiltonian, psi0, time):
    return expm_multiply(-1j * time * hamiltonian, psi0)


def construct_h_np(delta, h_list):
    if isinstance(h_list, (list, tuple)):
        hamiltonian = h_list[0].copy()
        for coeff, term in zip(delta, h_list[1:]):
            hamiltonian = hamiltonian + coeff * term
        return hamiltonian
    return h_list[0] + np.einsum("k,kij->ij", delta, h_list[1:])


def make_initial_state(config, n):
    state_type = config["type"]
    if state_type == "computational_basis":
        return computational_basis_state(config["label"], n)
    if state_type == "plus":
        return plus_state(n)
    if state_type == "ghz":
        return ghz_state(n)
    if state_type == "random_product":
        return random_product_state(n, seed=config.get("seed", 0))
    if state_type == "chaotic_evolved":
        return chaotic_evolved_state(config, n)
    raise ValueError(f"Unknown initial state type: {state_type}")


def computational_basis_state(label, n):
    if len(label) != n:
        raise ValueError(f"Basis label length {len(label)} does not match n={n}")
    index = int(label, 2)
    state = np.zeros(2**n, dtype=complex)
    state[index] = 1.0
    return jnp.asarray(state)


def plus_state(n):
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    return jnp.asarray(state)


def ghz_state(n):
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1 / np.sqrt(2)
    state[-1] = 1 / np.sqrt(2)
    return jnp.asarray(state)


def random_product_state(n, seed=0):
    rng = np.random.default_rng(seed)
    state = np.asarray([1.0 + 0.0j])
    for _ in range(n):
        theta = np.arccos(1 - 2 * rng.random())
        phi = 2 * np.pi * rng.random()
        qubit = np.asarray(
            [
                np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2),
            ],
            dtype=complex,
        )
        state = np.kron(state, qubit)
    return jnp.asarray(state)


def chaotic_evolved_state(config, n):
    base_state = np.asarray(make_initial_state(config["base_state"], n))
    hamiltonian = Nearest_Neighbour_1d(
        n=n,
        **config["hamiltonian"]["params"],
    )
    evolved = expH(hamiltonian.ham, config["time"]).data @ base_state
    return jnp.asarray(evolved)


def quadratic_error(deltas, qfim):
    values = np.einsum("...i,ij,...j->...", deltas, qfim, deltas)
    return 0.5 * np.sqrt(np.maximum(values, 0.0))


def safe_ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 1e-15,
    )


def fit_power_law(x, y):
    mask = np.isfinite(y) & (y > 0)
    if np.count_nonzero(mask) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)
    return slope
