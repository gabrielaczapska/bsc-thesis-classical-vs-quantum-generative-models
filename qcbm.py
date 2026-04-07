import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import optax
from scipy.special.cython_special import kl_div

np.random.seed(81)


jax.config.update("jax_enable_x64", True)

class MMD:
    def __init__(self, scales, space):
        self.scales = scales
        gammas = 1 / (2 * (self.scales**2))
        sq_dists = jnp.abs(space[:, None] - space[None, :] ** 2)
        # measure the similarities between the datapoints considering different gamma values (level of detail)
        self.K = sum(jnp.exp(-gammas[None, :] * sq_dists) for gamma in gammas) / len(scales)

    def k_expval(self, px, py):
        # kernel expectation value - average similarity across px and py distributions considering different level of similarities (self.K)
        return px @ self.K @ py

    def __call__(self, px, py):
        pxy = px - py
        return self.k_expval(pxy, pxy)


class QCBM:
    def __init__(self, circuit, mmd, py):
        # circuit - quantum algorithm expressed with quantum gates
        self.circuit = circuit
        self.mmd = mmd
        self.py = py

    @partial(jax.jit, static_argnums=0)
    def mmd_loss(self, params):
        px = self.circuit(params)
        return self.mmd(px, self.py), px


def represent_as_integers(data):
    bitstrings = []
    nums = []

    for d in data:
        bitstrings += ["".join(str(int(i))) for i in d]
        nums += [int(bitstrings[-1], 2)]
    # output: [0, 292, 146, 438, 73, 365, 219, 448, 56, 504, 7, 455, 63, 511]
    return bitstrings, nums


def define_and_visualise_target_distributions(size, data):
    # assign probabilities to each of 512 patterns that can be defined on 9 pixels
    probs = jnp.zeros(2**size)
    bitstrings, nums = represent_as_integers(data)
    probs[nums] = 1 / len(data)

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(nums)), probs, width=2.0, label=r"$\pi(x)$")
    plt.xticks(nums, bitstrings, rotation=45)

    plt.xlabel("Pattern")
    plt.ylabel("Probability Distribution")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    return probs


def q_circuit_specifications(size):
    # np.random.seed(42)
    n_qubits = size
    #device = qml.device("default.qubit", wires=n_qubits)

    # nr of layers (the rows of parallel qubits) ~ depth; the more layers -> the higher the "expressibility"
    n_layers = 6
    wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    # randomise the matrix of weights in the circuit; it defines the rotation angles of the quantum gates
    weights = np.random.random(size=wshape)
    return weights, n_layers, n_qubits, wshape

@qml.qnode(qml.device("default.qubit", wires=9))
def circuit(size):
    weights, n_layers, n_qubits, _ = q_circuit_specifications(size)
    qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))


jit_circuit = jax.jit(circuit)


# INITIALISATION
bandwidth = jnp.array([0.25, 0.5, 1])
# n_qubits -> 9
space = jnp.arange(2 ** 9)
mmd = MMD(bandwidth, space)
# toDo correctly define size and data
qcbm = QCBM(jit_circuit, mmd, define_and_visualise_target_distributions(size, data))

weights, _, _, _ = circuit(size)
opt = optax.adam(learning_rate=0.01)
# opt_state - stores the internal state of the optimiser
opt_state = opt.init(weights)

# LOSSES CALCULATION
# --
# fill in
# --


# TRAINING
# calculate loss, generated distribution and gradients that are used to set the direction and change the state of the opt
# calculate the differences across the distributions with kl_div
@jax.jit
def update_step(params, opt_state):
    (loss_value, qcbm_probs), grads = jax.value_and_grad(qcbm.mmd_loss, has_aux=True)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    log_ratio = jnp.where(qcbm.py == 0, 0, jnp.log(qcbm_probs / qcbm.py))

    kl_div = -jnp.sum(qcbm.py * log_ratio)

    return params, opt_state, loss_value, kl_div


def trace_100_update_steps(params, opt_state):
    history = []
    divergences = []
    n_iterations = 100

    for i in range(n_iterations):
        weights, opt_state, loss_value, kl_div = update_step(params, opt_state)

        if i % 10 == 0:
            print(f"Step: {i}, Loss: {loss_value:.4f}, KL-divergence: {kl_div:.4f}")

        history.append(loss_value)
        divergences.append(kl_div)

    return history, divergences


# TRAINING VISUALISATION
def visualise_training(params, opt_state):
    history, divergences = trace_100_update_steps(params, opt_state)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("MMD Loss")

    ax[1].plot(divergences)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("KL Divergence")

    plt.show()


# COMPARISON OF GENERATED AND TARGET DISTRIBUTIONS
# generated patterns are based on the weights (gate rotation angles); with probabilities being the output of the circuit
def comparison(weights, size, probs, nums, bitstrings):
    qcbm_probs = np.array(qcbm.circuit(weights))

    plt.figure(figsize=(12, 5))

    plt.bar(
        np.arange(2**size),
        probs,
        width=2.0,
        label=r"$\pi(x)$",
        alpha=0.4,
        color="tab:green",
    )

    plt.bar(
        np.arange(2**size),
        qcbm_probs,
        width=2.0,
        label=r"$p_\theta(x)$",
        alpha=0.9,
        color="tab:red",
    )

    plt.xlabel("Samples")
    plt.ylabel("Probability Distribution")

    plt.xticks(nums, bitstrings, rotation=45)
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.3)
    plt.show()


# TESTING
def circuit(weights, n_layers, n_qubits):
    qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
    return qml.sample()


# compute the chance of the model to create something meaningful (chi)
for N in [2000, 20000]:
    device = qml.device("default.qubit", wires=n_qubits)
    circuit = qml.set_shots(qml.QNode(circuit, device), shots=N)
    predictions = circuit(weights)
    mask = np.any(np.all(predictions[:, None] == data, axis=2), axis=1)
    chi = np.sum(mask) / N
    print(f"Validity (chi) for N = {N}: {chi:.4f}")

print(f"Chi for N = ∞: {np.sum(qcbm_probs[nums]):.4f}")

## Visualisation of the patterns
plt.figure(figsize=(8, 8))
j = 1

for i, m in zip(preds[:64], mask[:64]):
    ax = plt.subplot(8, 8, j)
    j += 1
    plt.imshow(np.reshape(i, (n, n)), cmap="gray", vmin=0, vmax=1)
    if ~m:
        plt.setp(ax.spines.values(), color="red", linewidth=1.5)

    plt.xticks([])
    plt.yticks([])
