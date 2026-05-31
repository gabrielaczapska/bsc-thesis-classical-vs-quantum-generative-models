import time
from collections import Counter
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
import torch
from bars_and_stripes import *
from evaluation_wgan_gp import *
from plotting import *

# enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class MMD:
    """
    Maximum Mean Discrepancy (MMD) loss class.

    :param scales: kernel bandwidths
    :param space: 1D array enumerating all possible states (integers encoding bitstrings)
    """
    def __init__(self, scales, space):
        # convert bandwidths into Gaussian kernel parameters
        gammas = 1 / (2 * (scales**2))
        # compute pairwise squared distances between all possible states
        sq_dists = jnp.abs(space[:, None] - space[None, :]) ** 2
        # build an averaged Gaussian kernel matrix
        self.K = sum(jnp.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.scales = scales

    def k_expval(self, px, py):
        """
        Compute the kernel expectation between generated (px) and target (py) probability distributions.
        """
        return px @ self.K @ py

    def __call__(self, px, py):
        """
        Compute the MMD loss between two probability distributions: px and py.
        """
        pxy = px - py
        return self.k_expval(pxy, pxy)


class QCBM:
    def __init__(self, circ, mmd, py):
        """
        Quantum Circuit Born Machine model class.

        :param circ: quantum circuit returning probability distributions
        :param mmd: MMD loss object
        :param py: target probability distribution
        """
        self.circ = circ
        self.mmd = mmd
        self.py = py

    @partial(jax.jit, static_argnums=0)
    def mmd_loss(self, params):
        """
        Evaluate the MMD loss for the current circuit parameters.

        :param params: trainable and current circuit parameters
        :return: MMD loss and generated probability distribution (px)
        """
        px = self.circ(params)
        return self.mmd(px, self.py), px


def construct_circuit(n_qubits=9, n_layers=6):
    """
    Build a QCBM circuit that outputs full probability distribution.
    """
    # Initialise a simulator device
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights):
        # Apply a parameterised entangling ansatz
        qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
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

    :param params: current and trainable circuit parameters
    :param opt_state: current optimiser state
    :param opt: Optax optimiser
    :param qcbm: QCBM model
    :return: updated parameters, updated optimiser state, MMD loss, KL divergence
    """
    (loss_value, qcbm_probs), grads = jax.value_and_grad(qcbm.mmd_loss, has_aux=True)(params)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    log_ratio = jnp.where(qcbm.py == 0, 0, jnp.log(qcbm_probs / qcbm.py))
    kl_div = -jnp.sum(qcbm.py * log_ratio)

    return params, opt_state, loss_value, kl_div


def train(weights, opt_state, opt, qcbm, n_iterations=1500, visualise=False):
    """
    Train the QCBM.

    :param weights: initial circuit weights
    :param opt_state: initial optimiser state
    :param opt: Optax optimiser
    :param qcbm: QCBM model
    :param n_iterations: number of optimisation steps
    :param visualise: whether to plot training metrics
    :return: trained weights, loss history, KL divergence history
    """
    history = []
    divs = []

    start_time = time.time()

    print(f"Training for {n_iterations} iterations:")

    for i in range(n_iterations):
        weights, opt_state, loss_value, kl_div = update_step(weights, opt_state, opt, qcbm)

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

    :param n_qubits: number of qubits
    :param n_layers: number of entangling layers
    :return: PennyLane QNode (quantum circuit bound to a device) that returns samples
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


def total_variation_to_uniform_from_counts(counts, true_patterns, num_samples):
    """
    Compute TV distance from the empirical valid-pattern distribution to the target uniform distribution.

    :param counts: counts of generated patterns
    :param true_patterns: valid target patterns
    :param num_samples: total number of generated samples
    :return: TV distance and empirical probabilities over valid patterns
    """
    empirical = torch.tensor([counts.get(p, 0) / num_samples for p in true_patterns], dtype=torch.float32)
    uniform = torch.full_like(empirical, 1.0 / len(true_patterns))

    return 0.5 * torch.abs(empirical - uniform).sum().item(), empirical


def evaluate_qcbm_like_wgan(weights, target_patterns, n_qubits, n_layers, num_samples=5000):
    """
    Evaluate the QCBM using the same evaluation metrics as WGAN-GP models.
    """
    sampling_circuit = build_sampling_circuit(n_qubits, n_layers)
    circ = qml.set_shots(sampling_circuit, shots=num_samples)

    samples = np.asarray(circ(weights), dtype=int)

    true_patterns = [
        tuple(np.asarray(x, dtype=int).flatten().tolist())
        for x in target_patterns
    ]
    true_set = set(true_patterns)

    counts = Counter(tuple(s.tolist()) for s in samples)

    valid_counts = {k: v for k, v in counts.items() if k in true_set}
    invalid_counts = {k: v for k, v in counts.items() if k not in true_set}

    total_valid = sum(valid_counts.values())
    total_invalid = sum(invalid_counts.values())

    valid_ratio = total_valid / num_samples
    invalid_ratio = total_invalid / num_samples

    print(f"\nGenerated {num_samples} samples:")
    print(f"Covered valid patterns: {len(valid_counts)}/{len(true_set)}")
    print(f"Invalid unique patterns: {len(invalid_counts)}")

    print("\nOverall Quality:")
    print(f"Total valid samples:   {total_valid} ({valid_ratio:.4f})")
    print(f"Total invalid samples: {total_invalid} ({invalid_ratio:.4f})")


    tv, empirical = total_variation_to_uniform_from_counts(
        counts,
        true_patterns,
        num_samples,
    )

    print(f"\nTV distance to uniform over {len(true_patterns)} valid patterns: {tv:.4f}")

    print("\nEmpirical probs over valid support:")
    for i, p in enumerate(empirical.tolist(), start=1):
        print(f"pattern {i}: {p:.4f}")



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
    evaluate_qcbm_like_wgan(weights, data, n_qubits, n_layers, num_samples)
  #  compare_px_and_py(qcbm_probs, probs, nums, bitstrings)