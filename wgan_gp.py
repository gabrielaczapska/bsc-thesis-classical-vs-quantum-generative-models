import torch.nn as nn
import torch.optim as optim
from bars_and_stripes import *
from wgan_gp_training_losses import *
from wgan_gp_training import *
from evaluation import *


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

critic_batch_size = 64
generator_batch_size = 128

g_hidden = 64
c_hidden = 64

mod = 100

data_2d = make_bars_and_stripes(3).to(device=device, dtype=torch.float32)
data = data_2d.view(-1, 9).float()

# Generator
class Generator(nn.Module):
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


# Discriminator
class Critic(nn.Module):
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
        return self.net(x)

G = Generator(z_dim, g_hidden, x_dim).to(device)

C = Critic(x_dim, c_hidden).to(device)

g_opt = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.9))
c_opt = optim.Adam(C.parameters(), lr=lr_c, betas=(0.5, 0.9))


# Training
train(C, G, epochs, n_critic, data, critic_batch_size, generator_batch_size, z_dim, device, lambda_gp, lambda_struct,
          mod, lambda_div, lambda_uniform, c_opt, g_opt)


# Evaluation
support_coverage_stats(G, target_patterns=data, z_dim=z_dim, device=device, num_samples=5000)
total_variation_to_uniform(G, data, z_dim, device, num_samples=5000)
