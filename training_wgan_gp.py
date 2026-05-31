from bars_and_stripes import *
from losses_functions_wgan_gp import *
import time


def train(C, G, epochs, n_critic, data, critic_batch_size, generator_batch_size, z_dim, device, lambda_gp, lambda_struct, lambda_div, lambda_uniform, print_every, c_opt, g_opt):
    """
    Purpose:
    Train a WGAN-GP generator and discriminator on the bars-and-stripes dataset.

    The critic learns to assign higher scores to real samples than to generated ones, thereby separating the two distributions.
    The generator is trained to produce samples that receive high critic scores (i.e., are indistinguishable from real data),
    while also matching the structure and distribution of valid patterns.

    :param n_critic: number of critic updates per generator update
    :param z_dim: number of latent dimensions in the generator input
    :param lambda_gp: weight for the gradient penalty term
    :param lambda_struct: weight for the structure loss term
    :param print_every: interval (in epochs) at which training statistics are printed
    :param lambda_div: weight for the diversity loss term
    :param lambda_uniform: weight for the uniformity loss term
    :param c_opt: optimiser for the critic
    :param g_opt: optimiser for the generator
    :return: None, updates C and G in place and prints the training diagnostics
    """
    start_time = time.time()

    for epoch in range(epochs):
        # train critic
        for _ in range(n_critic):
            # sample real data from the target distribution
            real_samples = sample_real_batch(data, batch_size=critic_batch_size)

            # generate fake samples defined on z_dim latent dimensions
            z = torch.randn(real_samples.size(0), z_dim, device=device)
            fake_samples = G(z).detach()

            # critic scores for real and fake samples
            c_real = C(real_samples)
            c_fake = C(fake_samples)

            # estimate of Wasserstein distance: E[C(real)] - E[C(fake)]
            wasserstein_est = c_real.mean() - c_fake.mean()

            # gradient penalty enforces 1-Lipschitz constraint on the critic
            gp = gradient_penalty(C, real_samples, fake_samples)

            # critic loss (minimised) -
            c_loss = -wasserstein_est + lambda_gp * gp

            # reset gradients
            c_opt.zero_grad()
            c_loss.backward()
            c_opt.step()

        # train generator
        z = torch.randn(generator_batch_size, z_dim, device=device)
        fake_samples = G(z)

        # adversarial loss -
        adv_loss = -C(fake_samples).mean()
        struct_loss_value = structure_loss(fake_samples)
        div_loss_value = diversity_loss(fake_samples)
        uniform_loss_value = uniformity_loss(fake_samples, data, temperature=12.0)

        # combined generator objective, each loss is controlled by its own lambda coefficient
        g_loss = (
                adv_loss
                + lambda_struct * struct_loss_value
                + lambda_div * div_loss_value
                + lambda_uniform * uniform_loss_value
        )

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        # training diagnostics
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

