"""
Loss functions used for training WGAN-GP models on the 3x3 Bars and Stripes dataset.
"""

import torch


def structure_loss(fake_samples):
    """
    Encourage generated samples to form valid Bars and Stripes patterns, either horizontal stripes or vertical bars.

    This loss measures variation across rows and columns in generated 3x3 samples.
    Valid patterns have either low row variation (stripes) or low column variation (bars).

    :param fake_samples: Tensor containing generated samples
    :return: Scalar structure loss
    """
    fake_samples = fake_samples.view(-1, 3, 3)

    # Row-wise and column-wise variance
    row_var = fake_samples.var(dim=2).mean()
    col_var = fake_samples.var(dim=1).mean()

    # Encourage either horizontal or vertical consistency
    return torch.minimum(row_var, col_var)


def diversity_loss(fake_samples):
    """
    Encourage the production of diverse samples within a batch.

    This loss discourages mode collapse by penalising batches where generated samples are too similar to one another.

    :param fake_samples: Tensor containing generated samples
    :return: Scalar diversity loss
    """
    batch_size = fake_samples.size(0)

    # Pairwise mean absolute differences between samples
    diffs = torch.abs(fake_samples.unsqueeze(1) - fake_samples.unsqueeze(0)).mean(dim=2)

    # Ignore self-comparisons along the diagonal
    mask = 1 - torch.eye(batch_size, device=fake_samples.device)

    mean_dist = (diffs * mask).sum() / mask.sum()

    # Larger distances should reduce the loss
    return 1.0 / (mean_dist + 1e-8)


def uniformity_loss(fake_samples, target_patterns, temperature):
    """
    Encourage the generator to produce all valid patterns uniformly.

    This loss penalises deviations from a uniform distribution over target patterns.
    Each generated sample is matched against the set of valid target patterns using a temperature-scaled similarity measure.

    :param fake_samples: Tensor containing generated samples
    :param target_patterns: Tensor containing target patterns
    :param temperature: Controls the sharpness of pattern assignment,
                        with higher values focusing more strongly on a single pattern
    :return: Scalar uniformity loss
    """
    # Mean squared distances to all target patterns
    dists = ((fake_samples.unsqueeze(1) - target_patterns.unsqueeze(0)) ** 2).mean(dim=2)

    # Convert distances into soft assignment probabilities
    logits = -temperature * dists
    probs = torch.softmax(logits, dim=1)

    # Average assignment probability over the batch
    mean_probs = probs.mean(dim=0)

    # Ideal uniform target distribution
    target = torch.full_like(mean_probs, 1.0 / target_patterns.size(0))

    return ((mean_probs - target) ** 2).mean()


def gradient_penalty(critic, real_samples, fake_samples):
    """
    Compute the WGAN-GP gradient penalty.

    The penalty enforces the 1-Lipschitz constraint required by the Wasserstein critic
    by encouraging gradient norms close to 1 on samples interpolated between real and generated data.

    :param critic: Critic model
    :param real_samples: Tensor containing real samples
    :param fake_samples: Tensor containing generated samples
    :return: Scalar gradient penalty
    """
    batch_size = real_samples.size(0)

    # Random interpolation coefficients
    alpha = torch.rand(batch_size, 1, device=real_samples.device)
    alpha = alpha.expand_as(real_samples)

    # Interpolated samples between real and fake data
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    # Critic scores for interpolated samples
    critic_scores = critic(interpolates)

    # Compute gradients of critic outputs with respect to inputs
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)

    # Penalise deviation of gradient norm from 1
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    return gp