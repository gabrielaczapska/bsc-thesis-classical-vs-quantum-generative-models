import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from evaluation import *
from wgan_gp_training_losses import *
from bars_and_stripes import *
from wgan_gp_training import *


# Configuration
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
n_qubits = 6
g_q_layers = 2

critic_batch_size = 64
generator_batch_size = 128

# Classical critic settings
c_hidden = 64

mod = 100

data_2d = make_bars_and_stripes(3).to(device=device, dtype=torch.float32)
data = data_2d.view(-1, 9).float()


# Quantum generator simulator for the generator
g_dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(g_dev, interface="torch", diff_method="backprop")
def quantum_generator_circuit(z_angles, weights):
    """
    Quantum circuit used inside the generator.

    Input angles are encoded as qubit rotations, passed through trainable
    entangling layers, and measured to produce one feature value per qubit.

    :param z_angles: rotation angle vector, one angle per qubit
    :param weights: trainable parameters of the quantum circuit
    :return: list of Pauli-Z expectation values in [-1, 1], one per qubit
    """
    # Encode classical input angles as rotations on each qubit
    for i in range(n_qubits):
        qml.RY(z_angles[i], wires=i)

    # Apply strongly entangled quantum layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Measure each qubit in the Z-basis to obtain one feature per qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumGenerator(nn.Module):
    """
    Hybrid quantum-classical generator for bars-and-stripes samples.
    Combines classical layers with a quantum circuit.

    Flow:
    latent noise -> circuit angles -> quantum features -> flattened 3×3 image.
    """
    def __init__(self, z_dim, n_qubits, n_layers, x_dim):
        """
        Initialise the hybrid generator.

        :param z_dim: dimension of the latent noise vector
        :param n_qubits: number of qubits in the quantum circuit
        :param n_layers: number of trainable quantum layers
        :param x_dim: output size
        """
        super().__init__()

        self.z_dim = z_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.x_dim = x_dim

        # Convert random noise into quantum rotation angles
        self.noise_to_angles = nn.Linear(z_dim, n_qubits)

        # Trainable quantum parameters inside the quantum circuit
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3, dtype=torch.float32))

        # Classical postprocessing: map quantum outputs -> 64 hidden features -> ReLU -> 9 values (flattened 3×3 image)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, x_dim)
        )

    def forward(self, z):
        """
        Generate flattened image samples from latent noise.
        """
        z = z.to(dtype=torch.float32)

        # Convert latent noise into bounded circuit angles
        angles = self.noise_to_angles(z)
        angles = torch.tanh(angles) * torch.pi

        outputs = []

        # Run the quantum circuit once per sample in the batch
        for i in range(z.size(0)):
            q_out = quantum_generator_circuit(angles[i], self.q_weights)
            q_out = torch.stack(q_out).to(device=z.device, dtype=z.dtype)
            outputs.append(q_out)

        # Combine all quantum outputs into one batch
        q_out = torch.stack(outputs, dim=0)

        # Map quantum features into raw flattened image values
        x = self.post(q_out)

        # Squash outputs to [0, 1], interpretable as pixel probabilities
        return torch.sigmoid(4.0 * x)


# Classical Discriminator
class Critic(nn.Module):
    """
    Classical WGAN-GP critic that assigns a real-valued score to each sample.
    """
    def __init__(self, x_dim, hidden_dim):
        """
        Initialise the critic network.

        :param x_dim: input size, (9 for a flattened 3×3 image)
        :param hidden_dim: hidden layer size
        """
        super().__init__()

        # Small neural network that maps each sample to one critic score
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Compute critic scores for a batch of samples.
        """
        return self.net(x)



G = QuantumGenerator(z_dim=z_dim, n_qubits=n_qubits, n_layers=g_q_layers, x_dim=x_dim).to(device)

C = Critic(x_dim=x_dim, hidden_dim=c_hidden).to(device)

g_opt = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.9))
c_opt = optim.Adam(C.parameters(), lr=lr_c, betas=(0.5, 0.9))


# Training
train(C, G, epochs, n_critic, data, critic_batch_size, generator_batch_size, z_dim, device, lambda_gp, lambda_struct,
          mod, lambda_div, lambda_uniform, c_opt, g_opt)


# Evaluation
support_coverage_stats(G, target_patterns=data, z_dim=z_dim, device=device, num_samples=5000)
total_variation_to_uniform(G, data, z_dim, device, num_samples=5000)
