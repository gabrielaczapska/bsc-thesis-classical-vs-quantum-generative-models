"""
Training framework for WGAN-GP models on the 3x3 Bars and Stripes dataset.
"""

import time
import torch

from bars_and_stripes import sample_real_batch
from loss_functions_wgan_gp import (gradient_penalty, structure_loss, diversity_loss, uniformity_loss)


def train(C,
          G,
          epochs,
          n_critic,
          data,
          critic_batch_size,
          generator_batch_size,
          z_dim,
          device,
          lambda_gp,
          lambda_struct,
          lambda_div,
          lambda_uniform,
          print_every,
          c_opt,
          g_opt
    ):
    """
    Train a WGAN-GP generator and critic on the Bars and Stripes dataset.

    The critic learns to assign higher scores to real samples than to generated ones.
    The generator is trained to produce samples that receive high critic scores, while also
    matching the structure and distribution of valid Bars and Stripes patterns.

    :param C: Critic model
    :param G: Generator model
    :param epochs: Number of training epochs
    :param n_critic: Number of critic updates per generator update
    :param data: Tensor containing real training samples
    :param critic_batch_size: Batch size used for critic updates
    :param generator_batch_size: Batch size used for generator updates
    :param z_dim: Dimension of the latent noise vector
    :param device: Device used for computation
    :param lambda_gp: Weight for the gradient penalty term
    :param lambda_struct: Weight for the structure loss term
    :param lambda_div: Weight for the diversity loss term
    :param lambda_uniform: Weight for the uniformity loss term
    :param print_every: Interval at which training diagnostics are printed
    :param c_opt: Optimiser for the critic
    :param g_opt: Optimiser for the generator
    :return: None, updates C and G in place and prints training diagnostics
    """
    start_time = time.time()

    for epoch in range(epochs):
        c_loss = gp = wasserstein_est = 0.0

        # Critic update step
        for _ in range(n_critic):
            real_samples = sample_real_batch(data, batch_size=critic_batch_size)

            # Latent vectors from a standard normal distribution
            z = torch.randn(real_samples.size(0), z_dim, device=device)

            # Fake samples generated without updating generator gradients
            fake_samples = G(z).detach()

            # Critic scores for real and generated samples
            c_real = C(real_samples)
            c_fake = C(fake_samples)

            # Wasserstein distance estimate: E[C(real)] - E[C(fake)]
            wasserstein_est = c_real.mean() - c_fake.mean()

            # Gradient penalty encouraging the 1-Lipschitz constraint
            gp = gradient_penalty(C, real_samples, fake_samples)

            # Critic loss: negative Wasserstein objective with gradient penalty
            c_loss = -wasserstein_est + lambda_gp * gp

            # Reset of stored gradients, backpropagation, and parameter update
            c_opt.zero_grad()
            c_loss.backward()
            c_opt.step()

        # Generator update step
        z = torch.randn(generator_batch_size, z_dim, device=device)
        fake_samples = G(z)

        # Wasserstein adversarial loss term
        adv_loss = -C(fake_samples).mean()

        # Auxiliary generator loss terms
        struct_loss_value = structure_loss(fake_samples)
        div_loss_value = diversity_loss(fake_samples)
        uniform_loss_value = uniformity_loss(fake_samples, data, temperature=12.0)

        # Total generator objective with weighted auxiliary losses
        g_loss = (
                adv_loss
                + lambda_struct * struct_loss_value
                + lambda_div * div_loss_value
                + lambda_uniform * uniform_loss_value
        )

        # Reset of stored gradients, backpropagation, and parameter update
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        # Training diagnostics
        if epoch % print_every == 0:
            print(
                f"Epoch {epoch:5d}, "
                f"C_loss={c_loss.item():.4f}, "
                f"G_loss={g_loss.item():.4f}, "
                f"W_est={wasserstein_est.item():.4f}, "
                f"GP={gp.item():.4f}, "
                f"adv={adv_loss.item():.4f}, "
                f"struct={struct_loss_value.item():.4f}, "
                f"div={div_loss_value.item():.4f}, "
                f"uniform={uniform_loss_value.item():.4f}"
            )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nUsing device: {device}")
    print(f"\nCritic Batch Size: {critic_batch_size}, Generator Batch Size: {generator_batch_size}")
    print(f"\nTraining time: {elapsed_time:.2f} seconds")
