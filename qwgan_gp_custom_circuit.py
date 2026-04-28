import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

from bars_and_stripes import *
from evaluation import *
from wgan_gp_training import *
from wgan_gp_training_losses import *


# Configurations
device = "cuda" if torch.cuda.is_available() else "cpu"

n = 3
x_dim = n * n
z_dim = 12

epochs = 15000

lr_g = 1e-4
lr_c = 1e-4

# WGAN-GP settings
n_critic = 5
lambda_gp = 10.0

# Loss weights for generator regularisers
lambda_struct = 2.0
lambda_div = 0.5
lambda_uniform = 4.0

# Quantum generator settings
n_qubits = 4
n_q_layers = 3

critic_batch_size = 64
generator_batch_size = 128

g_hidden = 64
c_hidden = 64

mod = 100

data_2d = make_bars_and_stripes(3).to(device)
data = data_2d.view(-1, 9).float()


# Quantum generator circuit
# Define PennyLane simulator device with n_qubits wires
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_generator_circuit(z_in, weights):
    """
    Quantum circuit used inside the generator.

    The input latent features are encoded as rotation angles. Trainable quantum
    layers are then applied, and Pauli-Z expectation values are returned as
    quantum features.

    :param z_in: latent angle vector of length n_qubits
    :param weights: trainable quantum circuit parameters
    :return: list of Pauli-Z expectation values, one per qubit
    """
    # Encode classical latent features into qubit rotations
    for i in range(n_qubits):
        qml.RY(z_in[i], wires=i)

    # Apply trainable quantum layers
    for layer in range(n_q_layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

        # Entangle neighbouring qubits
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

    # Return one expectation value per qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QGenerator(nn.Module):
    """
    Hybrid quantum-classical generator.

    A classical network maps latent noise to quantum circuit angles. The quantum circuit produces features,
    which are then mapped by another classical network to a generated 3x3 pattern.

    :param z_dim: dimension of the latent noise vector
    :param hidden_dim: hidden layer size for classical networks
    :param x_dim: output dimension of generated samples
    """
    def __init__(self, z_dim, hidden_dim, x_dim):
        super().__init__()

        # Trainable quantum circuit parameters
        self.weights = nn.Parameter(0.01 * torch.randn(n_q_layers, n_qubits, 2))

        # Map latent noise to n_qubits circuit input angles
        self.latent_proj = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_qubits))

        # Map quantum features to raw outputs before sigmoid
        self.head = nn.Sequential(nn.Linear(n_qubits, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, x_dim))

    def forward(self, z):
        """
        Generate samples from latent noise.
        """
        # Convert latent noise into circuit angles
        angles = self.latent_proj(z)           # [B, n_qubits]
        angles = torch.tanh(angles) * torch.pi

        # Run a single quantum circuit evaluation
        q_features = []
        for i in range(angles.size(0)):
            q_out = quantum_generator_circuit(angles[i], self.weights)
            q_out = torch.stack(q_out).float()

            # Convert Pauli-Z expectation values from [-1, 1] to [0, 1]
            q_out = (1.0 - q_out) / 2.0
            q_features.append(q_out)

        # Stack quantum features
        q_features = torch.stack(q_features, dim=0)

        # Convert quantum features into pattern probabilities
        x = self.head(q_features)

        return torch.sigmoid(2.0 * x)


class Critic(nn.Module):
    """
    WGAN critc.
    """
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Score a batch of patterns (x).
        """
        return self.net(x)


G = QGenerator(z_dim, g_hidden, x_dim).to(device)
C = Critic(x_dim, c_hidden).to(device)

g_opt = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.9))
c_opt = optim.Adam(C.parameters(), lr=lr_c, betas=(0.5, 0.9))


# Training
train(C, G, epochs, n_critic, data, critic_batch_size, generator_batch_size, z_dim, device, lambda_gp, lambda_struct, mod, lambda_div, lambda_uniform, c_opt, g_opt)


# Evaluation
support_coverage_stats(G, data, z_dim, device, num_samples=5000)
total_variation_to_uniform(G, data, z_dim, device, num_samples=5000)

