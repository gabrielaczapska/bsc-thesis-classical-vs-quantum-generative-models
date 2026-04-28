import torch

def structure_loss(fake_samples):
    """
    Purpose:
    Encourages the generated samples to form valid patterns: either horizontal stripes or vertical bars.

    :return: scalar loss (the lower, the better)
    """
    fake_samples = fake_samples.view(-1, 3, 3)

    row_var = fake_samples.var(dim=2).mean()
    col_var = fake_samples.var(dim=1).mean()

    return torch.minimum(row_var, col_var)

def diversity_loss(fake_samples):
    """
    Purpose:
    Encourages the generator to produce DIFFERENT samples within a batch, discourages mode collapse.

    :return: scalar loss (the lower, the better)
    """
    batch_size = fake_samples.size(0)

    diffs = torch.abs(fake_samples.unsqueeze(1) - fake_samples.unsqueeze(0)).mean(dim=2)

    mask = 1 - torch.eye(batch_size, device=fake_samples.device)
    mean_dist = (diffs * mask).sum() / mask.sum()

    return 1.0 / (mean_dist + 1e-8)

def uniformity_loss(fake_samples, target_patterns, temperature):
    """
    Purpose:
    Encourages the generator to produce ALL valid patterns evenly.

    :param temperature: controls whether a sample is matched to a single pattern or spread across many;
    higher values make the matching focus more on one pattern
    :return: scalar loss (the lower, the better)
    """
    dists = ((fake_samples.unsqueeze(1) - target_patterns.unsqueeze(0)) ** 2).mean(dim=2)
    logits = -temperature * dists
    probs = torch.softmax(logits, dim=1)

    mean_probs = probs.mean(dim=0)
    target = torch.full_like(mean_probs, 1.0 / target_patterns.size(0))

    return ((mean_probs - target) ** 2).mean()


def gradient_penalty(critic, real_samples, fake_samples):
    """
    Purpose:
    Stabilises WGAN-GP training by enforcing a smooth critic function (Lipschitz constraint),
    ensuring the critic does not change too abruptly for small input changes.

    :return: scalar penalty (the lower, the better)
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=real_samples.device)
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    critic_scores = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp