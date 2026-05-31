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
from evaluation import *
from graphs import *

# keep 32-bit precision
jax.config.update("jax_enable_x64", False)


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


def evaluate_qcbm_like_wgan(weights, target_patterns, n_qubits, n_layers, num_samples=5000,
    top_k_invalid=10, show_flat_patterns=False):
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

    chi_metric = chi_metric_to_uniform(counts, true_patterns, num_samples)
    print(f"Chi metric to uniform valid distribution: {chi_metric:.6f}")

    tv, empirical = total_variation_to_uniform_from_counts(
        counts,
        true_patterns,
        num_samples,
    )

    print(f"\nTV distance to uniform over {len(true_patterns)} valid patterns: {tv:.4f}")

    print("\nEmpirical probs over valid support:")
    for i, p in enumerate(empirical.tolist(), start=1):
        print(f"pattern {i}: {p:.4f}")

    kl_metric = kl_divergence_to_uniform(valid_counts, true_patterns)
    print(f"\nKL divergence over valid support: {kl_metric:.6f}")

    if show_flat_patterns:
        print("\nValid Pattern Frequencies:")
        for i, pattern in enumerate(sorted(true_set), start=1):
            c = valid_counts.get(pattern, 0)
            p = c / num_samples
            print(f"\nPattern {i}: count={c}, prob={p:.4f}")
            print(torch.tensor(pattern))

        if invalid_counts:
            print("\nInvalid Patterns (top by frequency):")
            for i, (pattern, c) in enumerate(sorted(invalid_counts.items(), key=lambda x: -x[1])[:top_k_invalid], start=1):
                print(f"\nInvalid {i}: count={c}, prob={c / num_samples:.4f}")
                print(torch.tensor(pattern))


def compute_chi_shots(weights, data, n_qubits, n_layers, n_shots):
    """
    Estimate chi using finite-shot sampling.
    Chi is the fraction of generated samples that are valid target patterns.
    """
    s_circuit = build_sampling_circuit(n_qubits, n_layers)
    circ = qml.set_shots(s_circuit, shots=n_shots)

    preds = circ(weights)
    mask = np.any(np.all(preds[:, None] == data, axis=2), axis=1)

    return np.sum(mask) / n_shots, preds, mask


def compute_chi_exact(qcbm_probs, nums):
    """
    Compute exact chi from the full generated probability distribution.
    """
    return np.sum(qcbm_probs[nums])


def evaluate_chi(weights, data, qcbm_probs, nums, n_qubits, n_layers, shot_counts=(2000, 20000), visualise=False):
    """
    Evaluate chi using both finite-shot sampling and exact probabilities.
    """
    data = np.asarray(data, dtype=float)

    print("\nEvaluating Chi:")
    for n_shots in shot_counts:
        chi, preds, mask = compute_chi_shots(weights, data, n_qubits, n_layers, n_shots)
        print(f"χ for N = {n_shots}: {chi:.4f}")

        if visualise:
            mark_invalid_patterns(preds, mask, N=n_shots)

    print(f"χ for N = ∞: {compute_chi_exact(qcbm_probs, nums):.4f}")


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
    evaluate_qcbm_like_wgan(weights, data, n_qubits, n_layers, num_samples, show_flat_patterns=False)
    evaluate_chi(weights, data, qcbm_probs, nums, n_qubits, n_layers, visualise=True)
    compare_px_and_py(qcbm_probs, probs, nums, bitstrings)
