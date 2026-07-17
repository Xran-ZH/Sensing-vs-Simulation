import jax
import jax.numpy as jnp
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from jax.scipy.linalg import expm


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
