"""
Train and evaluate a hybrid quantum-classical WGAN-GP model with StronglyEntanglingLayers
on the 3x3 Bars and Stripes dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

from bars_and_stripes import make_bars_and_stripes
from training_wgan_gp import train
from evaluation_wgan_gp import evaluate_generated_distribution, total_variation


# Configurations
device = "cuda" if torch.cuda.is_available() else "cpu"

n = 3
x_dim = n * n
z_dim = 12

epochs = 15000

lr_g = 1e-4
lr_c = 1e-4

# Critic settings
n_critic = 5
lambda_gp = 10.0

# Loss weights for generator regularisers
lambda_struct = 2.0
lambda_div = 0.5
lambda_uniform = 4.0

# Quantum generator settings
n_qubits = 6
q_layers = 2

critic_batch_size = 64
generator_batch_size = 128

g_hidden = 64
c_hidden = 64

print_every = 100

# Dataset in n x n grid form
data_2d = make_bars_and_stripes(n).to(device=device, dtype=torch.float32)

# Flattened dataset for neural network input
data = data_2d.view(-1, x_dim).float()

# Quantum simulator for the generator quantum circuit
g_dev = qml.device("default.qubit", wires=n_qubits, shots=None)


@qml.qnode(g_dev, interface="torch", diff_method="backprop")
def quantum_generator_circuit(z_angles, weights):
    """Quantum circuit used inside the hybrid generator."""

    # Encode classical input angles as rotations on each qubit
    for i in range(n_qubits):
        qml.RY(z_angles[i], wires=i)

    # Apply strongly entangled quantum layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Measure each qubit in the Z-basis to obtain one feature per qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumGenerator(nn.Module):
    """Hybrid quantum-classical generator for producing Bars and Stripes patterns.

    The generator maps latent noise to quantum circuit rotation angles,
    extracts expectation-value features from the quantum circuit,
    and classically post-processes them into flattened 3x3 samples.
    """

    def __init__(self, z_dim, n_qubits, n_layers, hidden_dim, x_dim):
        """
        Initialise the hybrid generator.

        :param z_dim: Dimensions of the latent noise vector
        :param n_qubits: Number of qubits in the quantum circuit
        :param n_layers: Number of trainable quantum layers
        :param hidden_dim: Hidden layer size of the classical post-processing network
        :param x_dim: Output size
        """
        super().__init__()

        # Map latent noise to quantum circuit rotation angles
        self.noise_to_angles = nn.Linear(z_dim, n_qubits)

        # Trainable parameters of the variational quantum circuit
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
        )

        # Classical post-processing network
        self.post = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )

    def forward(self, z):
        """Generate flattened image samples from latent noise."""
        z = z.to(dtype=torch.float32)

        # Convert latent noise into bounded circuit angles
        angles = self.noise_to_angles(z)
        angles = torch.tanh(angles) * torch.pi

        outputs = []

        # Run the quantum circuit once per sample in the batch
        for i in range(z.size(0)):
            q_values = quantum_generator_circuit(angles[i], self.q_weights)
            q_sample = torch.stack(q_values).to(device=z.device, dtype=z.dtype)
            outputs.append(q_sample)

        # Combine all quantum outputs into one batch
        q_batch = torch.stack(outputs, dim=0)

        # Map quantum features into raw flattened image values
        x = self.post(q_batch)

        # Squash outputs to [0, 1], interpretable as pixel probabilities
        return torch.sigmoid(4.0 * x)


# Classical Critic
class Critic(nn.Module):
    """Critic network used in the WGAN-GP framework."""

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
        """Compute critic scores for a batch of samples."""
        return self.net(x)


if __name__ == "__main__":
    G = QuantumGenerator(z_dim=z_dim, n_qubits=n_qubits, n_layers=q_layers, hidden_dim=g_hidden, x_dim=x_dim).to(device)

    C = Critic(x_dim=x_dim, hidden_dim=c_hidden).to(device)

    g_opt = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.9))
    c_opt = optim.Adam(C.parameters(), lr=lr_c, betas=(0.5, 0.9))

    # Training
    train(C=C,
          G=G,
          epochs=epochs,
          n_critic=n_critic,
          data=data,
          critic_batch_size=critic_batch_size,
          generator_batch_size=generator_batch_size,
          z_dim=z_dim,
          device=device,
          lambda_gp=lambda_gp,
          lambda_struct=lambda_struct,
          lambda_div=lambda_div,
          lambda_uniform=lambda_uniform,
          print_every=print_every,
          c_opt=c_opt,
          g_opt=g_opt)

    # Evaluation
    evaluate_generated_distribution(G=G, target_patterns=data, z_dim=z_dim, device=device, num_samples=5000)
    total_variation(G=G, target_patterns=data, z_dim=z_dim, device=device, num_samples=5000)
