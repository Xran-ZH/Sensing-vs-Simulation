import copy
import json
from pathlib import Path

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from jax.scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


class Nearest_Neighbour_1d:
    """Nearest-neighbour 1D spin Hamiltonian.

    H = sum_i (hx X_i + hy Y_i + hz Z_i)
        + sum_i (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1})
    """

    def __init__(self, n, Jx=0.0, Jy=0.0, Jz=0.0, hx=0.0, hy=0.0, hz=0.0):
        self.n = n
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.ham = self._build_hamiltonian()

    def _build_hamiltonian(self):
        labels = []
        coeffs = []

        for coeff, label in ((self.hx, "X"), (self.hy, "Y"), (self.hz, "Z")):
            for i in range(self.n):
                pauli = ["I"] * self.n
                pauli[i] = label
                labels.append("".join(pauli))
                coeffs.append(coeff)

        for coeff, label in ((self.Jx, "X"), (self.Jy, "Y"), (self.Jz, "Z")):
            for i in range(self.n - 1):
                pauli = ["I"] * self.n
                pauli[i] = label
                pauli[i + 1] = label
                labels.append("".join(pauli))
                coeffs.append(coeff)

        return SparsePauliOp(labels, coeffs=np.asarray(coeffs, dtype=complex))


def expH(hamiltonian, time):
    """Exact unitary exp(-i * H * time) as a Qiskit Operator."""
    matrix = _to_matrix(hamiltonian)
    return Operator(np.asarray(expm(-1j * time * matrix)))


def product_formula(hamiltonian, time, reps=1, order=1):
    """Approximate exp(-i * H * time) with a product formula.

    Args:
        hamiltonian: SparsePauliOp, matrix, or a list of Hamiltonian terms.
        time: Evolution time.
        reps: Number of Trotter steps.
        order: 1 for Lie-Trotter, 2 for symmetric Suzuki-Trotter.
    """
    if reps < 1:
        raise ValueError("reps must be at least 1")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    terms = _terms(hamiltonian)
    dt = time / reps
    step = _first_order_step(terms, dt)
    if order == 2:
        step = _second_order_step(terms, dt)

    dim = step.shape[0]
    unitary = jnp.eye(dim, dtype=complex)
    for _ in range(reps):
        unitary = step @ unitary
    return Operator(np.asarray(unitary))


def commutator(a, b):
    """Matrix commutator [A, B]."""
    a_matrix = _to_matrix(a)
    b_matrix = _to_matrix(b)
    return a_matrix @ b_matrix - b_matrix @ a_matrix


def norm(operator, ord=2):
    """Matrix norm helper used by simple product-formula estimates."""
    return jnp.linalg.norm(_to_matrix(operator), ord=ord)


def QFI_single_variable(h0, psi, theta, time):
    evolution_operator = expH(h0.ham, time * theta)
    psi_t = jnp.asarray(psi.evolve(evolution_operator).data)
    h0_matrix = _to_matrix(h0.ham)
    h0_squared = h0_matrix @ h0_matrix
    expectation_h = psi_t.conj().T @ h0_matrix @ psi_t
    expectation_h2 = psi_t.conj().T @ h0_squared @ psi_t
    return 4 * time**2 * (expectation_h2 - expectation_h**2).real


def Simulation_error_single_variable(delta, h0, psi, time):
    evolution_operator = expH(h0.ham, time * delta)
    psi_t = jnp.asarray(psi.evolve(evolution_operator).data)
    psi0 = jnp.asarray(psi.data)
    overlap = psi0.conj().T @ psi_t
    return pure_state_bures_distance_from_overlap(overlap)


def construct_H(delta, H_list):
    """Construct H(delta) = H_0 + sum_k delta_k H_k."""
    h_list = jnp.asarray(H_list)
    return h_list[0] + jnp.einsum("k,kij->ij", delta, h_list[1:])


@jax.jit
def state(delta, psi0, H_list, time):
    hamiltonian = construct_H(delta, H_list)
    return expm(-1j * time * hamiltonian) @ psi0


@jax.jit
def derivative(delta, psi0, H_list, time):
    y = state(delta, psi0, H_list, time)
    dy = jax.jacfwd(lambda d: state(d, psi0, H_list, time))(delta)
    return y, dy


@jax.jit
def core(y, dy):
    A = dy.conj().T @ dy
    B = dy.conj().T @ y
    return 4 * (A - jnp.outer(B, B.conj())).real


def General_QFI(delta, psi0, H_list, time):
    y, dy = derivative(delta, psi0, H_list, time)
    return core(y, dy)


def General_Simulation_error(delta, psi0, H_list, time):
    real = state(delta, psi0, H_list, time)
    ideal = state(jnp.zeros_like(delta), psi0, H_list, time)
    overlap = ideal.conj().T @ real
    return pure_state_bures_distance_from_overlap(overlap)


def pure_state_bures_distance_from_overlap(overlap):
    """Bures distance sqrt(2 * (1 - |<psi|phi>|)) for pure states."""
    fidelity_sqrt = jnp.minimum(jnp.abs(overlap), 1.0)
    return jnp.sqrt(jnp.maximum(0.0, 2.0 * (1.0 - fidelity_sqrt)))


def _first_order_step(terms, dt):
    step = jnp.eye(terms[0].shape[0], dtype=complex)
    for term in terms:
        step = expm(-1j * dt * term) @ step
    return step


def _second_order_step(terms, dt):
    step = jnp.eye(terms[0].shape[0], dtype=complex)
    for term in terms:
        step = expm(-0.5j * dt * term) @ step
    for term in reversed(terms):
        step = expm(-0.5j * dt * term) @ step
    return step


def _terms(hamiltonian):
    if isinstance(hamiltonian, SparsePauliOp):
        return [jnp.asarray(term.to_matrix(), dtype=complex) for term in hamiltonian]
    if isinstance(hamiltonian, (list, tuple)):
        return [_to_matrix(term) for term in hamiltonian]
    return [_to_matrix(hamiltonian)]


def _to_matrix(operator):
    if isinstance(operator, SparsePauliOp):
        return jnp.asarray(operator.to_matrix(), dtype=complex)
    if isinstance(operator, Operator):
        return jnp.asarray(operator.data, dtype=complex)
    return jnp.asarray(operator, dtype=complex)


# ---------------------------------------------------------------------------
# Experiment configuration and reusable numerical runners
# ---------------------------------------------------------------------------
#
# The run_*.py files are intentionally small: they define experiment parameters
# and call the routines below.  Keeping the numerical work here makes the
# convention for states, Hamiltonians, data files, QFI predictions, and Bures
# errors reusable across notebooks and future experiments.

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
    """Build the JSON-serializable config saved beside each dataset."""
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
    """Create human-readable dataset folder names from physics choices."""
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
        return "__".join(["ising1d", slug(h0.get("name", "h0")), "trotter-error"])

    return "__".join(
        [
            "ising1d",
            slug(h0.get("name", "h0")),
            "-".join(
                slug(item.get("name", f"delta{index}"))
                for index, item in enumerate(perturbations)
            ),
            state_slug(initial_state, scan),
            scan_slug(scan),
        ]
    )


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
    """Return psi0 and [H0, H1, ...] as dense JAX arrays for QFIM code."""
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
    """Sparse version of [H0, H1, ...] for repeated SciPy expm_multiply calls."""
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
    """Run a one-dimensional scan delta = epsilon * direction."""
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
    """Run a two-parameter scan and compare exact Bures error to QFIM ellipses."""
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
    """Compute directional QFI versus system size for several state families."""
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


def run_trotter_error(config, states=None):
    """Compare one-step product-formula errors to BCH-generator QFI predictions.

    The target Hamiltonian is split as H0 = A + B.  For order p=1,2 we use the
    leading BCH generator G_p and compare the exact one-step Bures distance
    between exp[-i(A+B) tau] and S_p(tau) with

        E_QFI = 1/2 sqrt(F_Q[G_p] * delta^2 * tau^(2p)).

    This routine is intentionally one-step: scanning `step_times` varies the
    single Trotter step size tau rather than the number of repeated steps.
    """
    n = config["n"]
    delta = float(config["scan"]["delta"])
    orders = np.asarray(config["scan"]["orders"], dtype=int)
    step_times = np.asarray(config["scan"]["step_times"], dtype=float)
    if states is None:
        states = {
            "plus": PLUS_STATE,
            "GHZ": GHZ_STATE,
            "random product": RANDOM_PRODUCT_STATE,
            "chaotic evolved": CHAOTIC_EVOLVED_STATE,
        }
    state_labels, state_vectors = make_state_bank(states, n)

    a_matrix, b_matrix = split_hamiltonian_terms(n, config["H0"]["params"])
    h_matrix = a_matrix + b_matrix
    generators = np.asarray(
        [bch_trotter_generator(a_matrix, b_matrix, order=int(order)) for order in orders]
    )

    trotter_error = np.zeros((len(orders), len(state_vectors), len(step_times)))
    qfi_effective = np.zeros_like(trotter_error)
    qfi_error = np.zeros_like(trotter_error)
    scaled_qfi = np.zeros_like(trotter_error)

    for order_index, (order, generator) in enumerate(zip(orders, generators)):
        h_list = jnp.stack([jnp.asarray(h_matrix), jnp.asarray(generator)])

        for time_index, tau in enumerate(step_times):
            exact_unitary, trotter_unitary = one_step_unitaries(
                h_matrix,
                [a_matrix, b_matrix],
                tau=float(tau),
                order=int(order),
            )
            amplitude = delta * tau**order

            for state_index, psi0 in enumerate(state_vectors):
                qfi = single_parameter_qfi(psi0, h_list, tau=float(tau))
                prediction = qfi * amplitude**2
                trotter_error[order_index, state_index, time_index] = (
                    bures_unitary_error(exact_unitary, trotter_unitary, psi0)
                )
                qfi_effective[order_index, state_index, time_index] = qfi
                scaled_qfi[order_index, state_index, time_index] = prediction
                qfi_error[order_index, state_index, time_index] = 0.5 * np.sqrt(
                    max(prediction, 0.0)
                )

    ratio = safe_ratio(trotter_error**2, qfi_error**2)
    output_dir = prepare_output_dir(config)
    np.savez(
        output_dir / config["results_file"],
        orders=orders,
        step_times=step_times,
        tau_values=step_times,
        delta=delta,
        state_labels=state_labels,
        trotter_error=trotter_error,
        leading_generators=generators,
        qfi_effective=qfi_effective,
        scaled_qfi=scaled_qfi,
        qfi_error=qfi_error,
        ratio=ratio,
    )
    return output_dir


def make_state_bank(states, n):
    state_items = list(states.items())
    labels = np.asarray([label for label, _ in state_items])
    vectors = [
        np.asarray(make_initial_state(state_config, n), dtype=complex)
        for _, state_config in state_items
    ]
    return labels, vectors


def one_step_unitaries(h_matrix, trotter_terms, tau, order):
    exact = np.asarray(expH(h_matrix, tau).data, dtype=complex)
    trotter = np.asarray(
        product_formula(trotter_terms, time=tau, reps=1, order=order).data,
        dtype=complex,
    )
    return exact, trotter


def bures_unitary_error(exact_unitary, approximate_unitary, psi0):
    exact_state = exact_unitary @ psi0
    approximate_state = approximate_unitary @ psi0
    return pure_state_bures_distance_np(np.vdot(exact_state, approximate_state))


def single_parameter_qfi(psi0, h_list, tau):
    qfim = np.asarray(General_QFI(jnp.zeros(1), jnp.asarray(psi0), h_list, tau))
    return directional_qfi(np.asarray([1.0]), qfim)


def split_hamiltonian_terms(n, params):
    """Split Nearest_Neighbour_1d parameters into onsite A and coupling B."""
    field_params = {
        "Jx": 0.0,
        "Jy": 0.0,
        "Jz": 0.0,
        "hx": params["hx"],
        "hy": params["hy"],
        "hz": params["hz"],
    }
    coupling_params = {
        "Jx": params["Jx"],
        "Jy": params["Jy"],
        "Jz": params["Jz"],
        "hx": 0.0,
        "hy": 0.0,
        "hz": 0.0,
    }
    a_matrix = np.asarray(
        Nearest_Neighbour_1d(n=n, **field_params).ham.to_matrix(),
        dtype=complex,
    )
    b_matrix = np.asarray(
        Nearest_Neighbour_1d(n=n, **coupling_params).ham.to_matrix(),
        dtype=complex,
    )
    return a_matrix, b_matrix


def bch_trotter_generator(a_matrix, b_matrix, order):
    """Leading BCH generator for one-step first/second order formulas."""
    comm_ab = commutator_np(a_matrix, b_matrix)
    if order == 1:
        generator = 0.5j * comm_ab
    elif order == 2:
        generator = (
            commutator_np(a_matrix, comm_ab) / 24.0
            + commutator_np(b_matrix, comm_ab) / 12.0
        )
    else:
        raise ValueError("Only order 1 and order 2 BCH generators are implemented.")
    return 0.5 * (generator + generator.conj().T)


def commutator_np(a_matrix, b_matrix):
    return a_matrix @ b_matrix - b_matrix @ a_matrix


def directional_qfi(direction, qfim):
    return float(np.einsum("i,ij,j->", direction, qfim, direction))


def simulation_error_np(delta, psi0, h_list, time, ideal_state):
    hamiltonian = construct_h_np(delta, h_list)
    real_state = evolve_state(hamiltonian, psi0, time)
    return pure_state_bures_distance_np(np.vdot(ideal_state, real_state))


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
    state = np.zeros(2**n, dtype=complex)
    state[int(label, 2)] = 1.0
    return jnp.asarray(state)


def plus_state(n):
    return jnp.asarray(np.ones(2**n, dtype=complex) / np.sqrt(2**n))


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
            [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
            dtype=complex,
        )
        state = np.kron(state, qubit)
    return jnp.asarray(state)


def chaotic_evolved_state(config, n):
    base_state = np.asarray(make_initial_state(config["base_state"], n))
    hamiltonian = Nearest_Neighbour_1d(n=n, **config["hamiltonian"]["params"])
    return jnp.asarray(expH(hamiltonian.ham, config["time"]).data @ base_state)


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
