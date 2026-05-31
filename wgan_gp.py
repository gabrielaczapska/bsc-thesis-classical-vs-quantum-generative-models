"""
Train and evaluate a WGAN-GP model on the 3x3 Bars and Stripes dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim

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

critic_batch_size = 64
generator_batch_size = 128

g_hidden = 64
c_hidden = 64

print_every = 100

# Dataset in n x n grid form
data_2d = make_bars_and_stripes(n).to(device=device, dtype=torch.float32)

# Flattened dataset for neural network input
data = data_2d.view(-1, x_dim)

# Generator
class Generator(nn.Module):
    """Generator network for producing Bars and Stripes patterns."""

    def __init__(self, z_dim, hidden_dim, x_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )

    def forward(self, z):
        x = self.net(z)
        return torch.sigmoid(4.0 * x)


# Critic
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
    G = Generator(z_dim=z_dim, hidden_dim=g_hidden, x_dim=x_dim).to(device)

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
