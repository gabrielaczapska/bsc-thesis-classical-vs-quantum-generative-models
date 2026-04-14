import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import pennylane as qml
import optax
from bars_and_stripes import *
from graphs import *

jax.config.update("jax_enable_x64", True)
np.random.seed(81)

class MMD:
    def __init__(self, scales, space):
        gammas = 1 / (2 * (scales**2))
        sq_dists = jnp.abs(space[:, None] - space[None, :]) ** 2
        # measure the similarities between the datapoints considering different gamma values (levels of details)
        self.K = sum(jnp.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.scales = scales

    def k_expval(self, px, py):
        # kernel expectation value - average similarity across px and py distributions given self.K
        return px @ self.K @ py

    def __call__(self, px, py):
        pxy = px - py
        return self.k_expval(pxy, pxy)


class QCBM:
    def __init__(self, circ, mmd, py):
        # quantum circuit expressed as a sequence of quantum gates
        self.circ = circ
        self.mmd = mmd
        self.py = py

    @partial(jax.jit, static_argnums=0)
    def mmd_loss(self, params):
        px = self.circ(params)
        return self.mmd(px, self.py), px


# Parametrised Quantum Circuit = Born Machine
def construct_circuit(n_qubits=9, n_layers=6):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights):
        qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
        return qml.probs()

    return circuit


def initialise_weights(n_qubits=9, n_layers=6):
    w_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return np.random.random(size=w_shape)
    #return jnp.array(np.random.random(size=w_shape))


# TRAINING
@partial(jax.jit, static_argnums=(2, 3))
def update_step(params, opt_state, opt, qcbm):
    (loss_value, qcbm_probs), grads = jax.value_and_grad(qcbm.mmd_loss, has_aux=True)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    log_ratio = jnp.where(qcbm.py == 0, 0, jnp.log(qcbm_probs / qcbm.py))
    kl_div = -jnp.sum(qcbm.py * log_ratio)

    return params, opt_state, loss_value, kl_div


def train(weights, opt_state, opt, qcbm, n_iterations=100, visualise=False):
    history = []
    divs = []

    print(f"Training for {n_iterations} iterations:")
    for i in range(n_iterations):
        weights, opt_state, loss_value, kl_div = update_step(weights, opt_state, opt, qcbm)

        if i % 10 == 0:
            print(f"Step: {i}, Loss: {loss_value:.4f}, KL-divergence: {kl_div:.4f}")

        history.append(loss_value)
        divs.append(kl_div)

    if visualise:
        plot_training_results(history, divs)

    return weights, history, divs


# # TESTING
def build_sampling_circuit(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def sampling_circuit(weights):
        qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
        return qml.sample()

    return sampling_circuit


def compute_chi_shots(weights, data, n_qubits, n_layers, n_shots):
    s_circuit = build_sampling_circuit(n_qubits, n_layers)
    circ = qml.set_shots(s_circuit, shots=n_shots)
    preds = circ(weights)
    mask = np.any(np.all(preds[:, None] == data, axis=2), axis=1)
    return np.sum(mask) / n_shots, preds, mask

def compute_chi_exact(qcbm_probs, nums):
    return np.sum(qcbm_probs[nums])


def evaluate_chi(weights, data, qcbm_probs, nums, n_qubits, n_layers, shot_counts=[2000, 20000], visualise=False):
    print("Evaluating Chi:")
    for N in shot_counts:
        chi, preds, mask = compute_chi_shots(weights, data, n_qubits, n_layers, N)
        print(f"χ for N = {N}: {chi:.4f}")
        if visualise:
            mark_invalid_patterns(preds, mask, N)
    print(f"χ for N = ∞: {compute_chi_exact(qcbm_probs, nums):.4f}")



if __name__=="__main__":
    n_qubits = 9
    n_layers = 6

    # probability distribution of the target patterns
    probs = define_and_visualise_target_distributions()

    data = get_bars_and_stripes(3)

    bitstrings, nums = represent_as_integers()

    # define bandwidth for scales def [MMD]
    bandwidth = jnp.array([0.25, 0.5, 1])
    space = jnp.arange(2 ** n_qubits)

    weights = initialise_weights(n_qubits, n_layers)
    circuit = construct_circuit(n_qubits, n_layers)
    jit_circuit = jax.jit(circuit)

    mmd = MMD(bandwidth, space)
    qcbm = QCBM(jit_circuit, mmd, probs)

    opt = optax.adam(learning_rate=0.1)
    opt_state = opt.init(weights)

    weights, history, divs = train(weights, opt_state, opt, qcbm, visualise=False)

    qcbm_probs = np.array(qcbm.circ(weights))

    evaluate_chi(weights, data, qcbm_probs, nums, n_qubits, n_layers, visualise=False)

    # compare the probability distributions of generated patterns to target patterns
    #compare_px_and_py(qcbm_probs, probs, nums, bitstrings)

    # visualise the QCBM
    #qml.draw_mpl(circuit, level="device")(weights)
    #plt.show()
