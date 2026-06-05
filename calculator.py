import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp, Statevector
from quantum_simulation_recipe.spin import Nearest_Neighbour_1d
from quantum_simulation_recipe.bounds import tight_bound, interference_bound, norm, commutator
from quantum_simulation_recipe.trotter import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

def QFI_single_variable(h0,psi,theta,time):
    Evolution_operator=expH(h0.ham,time*theta)
    psi_t=psi.evolve(Evolution_operator).data
    H0=h0.ham.to_matrix()
    H0_squared=H0@H0
    qfi=4*time**2*(psi_t.conj().T@H0_squared@psi_t- (psi_t.conj().T@H0@psi_t)**2).real
    return qfi

def Simulation_error_single_variable(delta, h0, psi, time):
    Evolution_operator=expH(h0.ham,time*delta)
    psi_t=psi.evolve(Evolution_operator).data
    return np.sqrt(1-abs(psi.data.conj().T@psi_t)**2)

def construct_H(delta,H_list):#delta is a vector of length k, H_list is a list of k+1 matrices, the first one is the base Hamiltonian, the rest are the perturbation Hamiltonians
    return H_list[0]+jnp.einsum("k,kij->ij",delta, H_list[1:])

@jax.jit
def state(delta, psi0, H_list,time): #psi0 is the initial state, H_list is a list of k+1 matrices, the first one is the base Hamiltonian, the rest are the perturbation Hamiltonians
    H = construct_H(delta, H_list)
    return expm(-1j*time*H)@psi0

@jax.jit
def derivative(delta, psi0, H_list,time):
    y = state(delta,psi0,H_list,time)
    dy = jax.jacfwd(lambda d: state(d,psi0,H_list,time))(delta) 
    return y,dy

@jax.jit
def core(y,dy):
    A = dy.conj().T @ dy
    B = dy.conj().T @ y
    return 4*(A - jnp.outer(B,B.conj())).real

def General_QFI(delta,psi0,H_list,time):
    y,dy = derivative(delta, psi0, H_list,time)
    return core(y,dy)

def General_Simulation_error(delta, psi0, H_list,time):
    Real = state(delta,psi0,H_list,time)
    ideal = state(jnp.zeros_like(delta),psi0,H_list,time)
    return jnp.sqrt(1-(ideal.conj().T@Real).real**2)