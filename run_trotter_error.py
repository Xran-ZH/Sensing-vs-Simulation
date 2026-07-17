import argparse

import jax.numpy as jnp
import numpy as np
from scipy.linalg import logm

from experiment_utils import (
    CHAOTIC_EVOLVED_STATE,
    DATA_DIR,
    DEFAULT_H0,
    GHZ_STATE,
    PLUS_STATE,
    RANDOM_PRODUCT_STATE,
    directional_qfi,
    make_config,
    make_initial_state,
    prepare_output_dir,
    pure_state_bures_distance_np,
)
from src import General_QFI, Nearest_Neighbour_1d, expH, product_formula


DATA_SUBDIR = "trotter_error"
EXPERIMENT_NAME = "trotter_error"

N = 6
TIME = 0.2
TARGET_H = DEFAULT_H0

SCAN = {
    "type": "trotter_error",
    "description": "Treat product-formula error as an effective Hamiltonian perturbation",
    "orders": [1, 2],
    "reps": [1, 2, 4],
}

STATES = {
    "plus": PLUS_STATE,
    "GHZ": GHZ_STATE,
    "random product": RANDOM_PRODUCT_STATE,
    "chaotic evolved": CHAOTIC_EVOLVED_STATE,
}


def build_config(smoke=False):
    scan = {
        **SCAN,
        "reps": [1, 2] if smoke else SCAN["reps"],
    }
    return make_config(
        experiment_name=EXPERIMENT_NAME,
        data_subdir=DATA_SUBDIR,
        n=3 if smoke else N,
        time=TIME,
        initial_state=PLUS_STATE,
        scan=scan,
        smoke=smoke,
        h0=TARGET_H,
        perturbations=[],
    )


def run_trotter_error(config):
    n = config["n"]
    time = config["time"]
    orders = np.asarray(config["scan"]["orders"], dtype=int)
    reps_values = np.asarray(config["scan"]["reps"], dtype=int)
    state_items = list(STATES.items())
    state_labels = np.asarray([label for label, _ in state_items])

    hamiltonian = Nearest_Neighbour_1d(n=n, **config["H0"]["params"]).ham
    h_matrix = np.asarray(hamiltonian.to_matrix(), dtype=complex)
    exact_unitary = np.asarray(expH(hamiltonian, time).data, dtype=complex)

    trotter_error = np.zeros((len(orders), len(state_items), len(reps_values)))
    qfi_effective = np.zeros_like(trotter_error)
    qfi_error = np.zeros_like(trotter_error)

    states = [
        np.asarray(make_initial_state(state_config, n), dtype=complex)
        for _, state_config in state_items
    ]
    exact_states = [exact_unitary @ psi0 for psi0 in states]

    for order_index, order in enumerate(orders):
        for reps_index, reps in enumerate(reps_values):
            trotter_unitary = np.asarray(
                product_formula(hamiltonian, time=time, reps=int(reps), order=int(order)).data,
                dtype=complex,
            )
            delta_h = effective_trotter_perturbation(trotter_unitary, h_matrix, time)
            h_list = jnp.stack([jnp.asarray(h_matrix), jnp.asarray(delta_h)])

            for state_index, psi0 in enumerate(states):
                trotter_state = trotter_unitary @ psi0
                overlap = np.vdot(exact_states[state_index], trotter_state)
                trotter_error[order_index, state_index, reps_index] = (
                    pure_state_bures_distance_np(overlap)
                )

                qfim = np.asarray(
                    General_QFI(
                        jnp.zeros(1),
                        jnp.asarray(psi0),
                        h_list,
                        time,
                    )
                )
                qfi_effective[order_index, state_index, reps_index] = directional_qfi(
                    np.asarray([1.0]), qfim
                )
                qfi_error[order_index, state_index, reps_index] = 0.5 * np.sqrt(
                    max(qfi_effective[order_index, state_index, reps_index], 0.0)
                )

    ratio = np.divide(
        trotter_error**2,
        qfi_error**2,
        out=np.full_like(trotter_error, np.nan),
        where=qfi_error > 1e-15,
    )

    output_dir = prepare_output_dir(config)
    np.savez(
        output_dir / config["results_file"],
        orders=orders,
        reps=reps_values,
        state_labels=state_labels,
        trotter_error=trotter_error,
        qfi_effective=qfi_effective,
        qfi_error=qfi_error,
        ratio=ratio,
    )
    return output_dir


def effective_trotter_perturbation(trotter_unitary, h_matrix, time):
    h_eff = (1j / time) * logm(trotter_unitary)
    h_eff = 0.5 * (h_eff + h_eff.conj().T)
    delta_h = h_eff - h_matrix
    return 0.5 * (delta_h + delta_h.conj().T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a tiny sanity-check dataset.")
    args = parser.parse_args()

    output_dir = run_trotter_error(build_config(smoke=args.smoke))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
