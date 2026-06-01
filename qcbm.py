"""
Train and evaluate a Quantum Circuit Born Machine (QCBM) on the 3x3 Bars and Stripes dataset.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml

from bars_and_stripes import make_bars_and_stripes, represent_as_integers
from plotting import *
from evaluation_wgan_gp import evaluate_qcbm, total_variation_qcbm


# 32-bit precision
jax.config.update("jax_enable_x64", False)


class MMD:
    """
    Maximum Mean Discrepancy (MMD) loss based on a Gaussian kernel.

    :param scales: Bandwidth parameters of the Gaussian kernel, controlling its smoothness
    :param space: 1D array of all possible integer representations of bitstrings
    """
    def __init__(self, scales, space):
        # Gaussian kernel parameters derived from bandwidths
        gammas = 1 / (2 * (scales**2))

        # Pairwise squared distances between all possible states
        sq_dists = jnp.abs(space[:, None] - space[None, :]) ** 2

        # Averaged Gaussian kernel matrix
        self.K = sum(jnp.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.scales = scales

    def k_expval(self, px, py):
        """Compute the kernel expectation between generated (px) and target (py) probability distributions."""
        return px @ self.K @ py

    def __call__(self, px, py):
        """Compute the MMD loss between two probability distributions: generated (px) and target (py)."""
        pxy = px - py
        return self.k_expval(pxy, pxy)


class QCBM:
    """Quantum Circuit Born Machine (QCBM) trained by minimising the MMD loss."""

    def __init__(self, circ, mmd, py):
        """
        Initialise the QCBM model.

        :param circ: Quantum circuit returning probability distributions
        :param mmd: MMD loss object
        :param py: Target probability distribution
        """
        self.circ = circ
        self.mmd = mmd
        self.py = py

    @partial(jax.jit, static_argnums=0)
    def mmd_loss(self, params):
        """
        Evaluate the MMD loss for the current circuit parameters.

        :param params: Current trainable circuit parameters
        :return: MMD loss and generated probability distribution (px)
        """
        px = self.circ(params)
        return self.mmd(px, self.py), px


def construct_circuit(n_qubits=9, n_layers=6):
    """
    Build a QCBM circuit returning the full probability distribution.
    """
    # Quantum simulator for the QCBM circuit
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights):
        # Trainable strongly entangling ansatz
        qml.StronglyEntanglingLayers(
            weights=weights,
            ranges=[1] * n_layers,
            wires=range(n_qubits)
        )
        return qml.probs()

    return circuit


def initialise_weights(n_qubits=9, n_layers=6):
    """
    Initialise random weights for the QCBM circuit.
    """
    w_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=n_layers,
        n_wires=n_qubits,
    )
    return np.random.random(size=w_shape)


@partial(jax.jit, static_argnums=(2, 3))
def update_step(params, opt_state, opt, qcbm):
    """
    Perform one optimisation step.

    :param params: Current trainable circuit parameters
    :param opt_state: Current optimiser state
    :param opt: Optax optimiser
    :param qcbm: QCBM model
    :return: Updated parameters, updated optimiser state, MMD loss, KL divergence
    """
    (loss_value, qcbm_probs), grads = jax.value_and_grad(
        qcbm.mmd_loss,
        has_aux=True
    )(params)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    log_ratio = jnp.where(qcbm.py == 0, 0, jnp.log(qcbm_probs / qcbm.py))
    kl_div = -jnp.sum(qcbm.py * log_ratio)

    return params, opt_state, loss_value, kl_div


def train(weights, opt_state, opt, qcbm, n_iterations=1500, visualise=False):
    """
    Train the QCBM by minimising the MMD loss.

    :param weights: Initial circuit weights
    :param opt_state: Initial optimiser state
    :param opt: Optax optimiser
    :param qcbm: QCBM model
    :param n_iterations: Number of optimisation steps
    :param visualise: Whether to plot training metrics
    :return: Trained weights, loss history, KL divergence history
    """
    history = []
    divs = []

    start_time = time.time()

    print(f"Training for {n_iterations} iterations:")

    for i in range(n_iterations):
        weights, opt_state, loss_value, kl_div = update_step(
            weights,
            opt_state,
            opt,
            qcbm
        )

        if i % 100 == 0:
            print(f"Step: {i}, MMD Loss: {loss_value:.4f}, KL-divergence: {kl_div:.4f}")

        history.append(loss_value)
        divs.append(kl_div)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTraining time: {elapsed_time:.2f} seconds")

    if visualise:
        plot_training_results(history, divs)

    return weights, history, divs


def build_sampling_circuit(n_qubits, n_layers):
    """
    Build a QCBM circuit that returns sampled bitstrings.

    :param n_qubits: Number of qubits
    :param n_layers: Number of entangling layers
    :return: PennyLane QNode returning sampled bitstrings when executed
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def sampling_circuit(weights):
        qml.StronglyEntanglingLayers(
            weights=weights,
            ranges=[1] * n_layers,
            wires=range(n_qubits),
        )
        # Return sampled bitstrings instead of exact probabilities
        return qml.sample()

    return sampling_circuit


if __name__ == "__main__":
    # Basic model configurations
    n_qubits = 9
    n_layers = 6
    num_samples = 5000

    # Construct the target probability distribution
    probs = define_and_visualise_target_distributions()
    data = make_bars_and_stripes(3)

    bitstrings, nums = represent_as_integers()

    bandwidth = jnp.array([0.25, 0.5, 1])
    space = jnp.arange(2 ** n_qubits)

    # Initialise model parameters and circuit
    weights = initialise_weights(n_qubits, n_layers)
    circuit = construct_circuit(n_qubits, n_layers)
    jit_circuit = jax.jit(circuit)

    # Build QCBM model
    mmd = MMD(bandwidth, space)
    qcbm = QCBM(jit_circuit, mmd, probs)

    # Define the optimiser
    opt = optax.adam(learning_rate=0.1)
    opt_state = opt.init(weights)

    # Training
    weights, history, divs = train(weights, opt_state, opt, qcbm)

    # Evaluation
    qcbm_probs = np.array(qcbm.circ(weights))
    sampling_circ = build_sampling_circuit(n_qubits, n_layers)
    evaluate_qcbm(weights, data, sampling_circ, num_samples)
    #compare_px_and_py(qcbm_probs, probs, nums, bitstrings)