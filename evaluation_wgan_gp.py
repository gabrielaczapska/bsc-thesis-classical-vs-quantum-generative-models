"""
Evaluation metrics for WGAN-GP models trained on the 3x3 Bars and Stripes dataset.
"""

import torch
from collections import Counter


def evaluate_generated_distribution(G, target_patterns, z_dim, device, num_samples, threshold=0.5):
    """
    Evaluate how well the generator G reproduces the target Bars and Stripes distribution.

    Report support validity, support coverage, and the empirical distribution over valid patterns.

    :param G: Generator model
    :param target_patterns: Tensor containing all target patterns
    :param z_dim: Dimensionality of the latent noise vector
    :param device: Device used for computation
    :param num_samples: Number of samples to generate
    :param threshold: Threshold for binarising generator outputs
    :return: None, prints evaluation statistics
    """
    G.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        samples = G(z)
        samples = (samples >= threshold).float().cpu()

    target_patterns = target_patterns.cpu()

    true_patterns = set(tuple(x.tolist()) for x in target_patterns)
    counts = Counter(tuple(s.tolist()) for s in samples)

    valid_counts = {
        pattern: count
        for pattern, count in counts.items()
        if pattern in true_patterns
    }

    invalid_counts = {
        pattern: count
        for pattern, count in counts.items()
        if pattern not in true_patterns
    }

    total_valid = sum(valid_counts.values())
    total_invalid = sum(invalid_counts.values())

    # Evaluation metrics
    validity_ratio = total_valid / num_samples
    invalidity_ratio = total_invalid / num_samples
    coverage_ratio = len(valid_counts) / len(true_patterns)

    valid_probability_distribution = {
        pattern: count / total_valid if total_valid > 0 else 0.0
        for pattern, count in valid_counts.items()
    }

    G.train()

    print(f"\nNumber of generated samples: {num_samples}")

    print("\nSupport Validity")
    print(f"Valid samples   : {total_valid} ({validity_ratio:.4f})")
    print(f"Invalid samples : {total_invalid} ({invalidity_ratio:.4f})")

    print("\nSupport Coverage")
    print(
        f"Covered valid patterns: "
        f"{len(valid_counts)}/{len(true_patterns)} ({coverage_ratio:.4f})"
    )

    print("\nEmpirical Distribution over Valid Patterns")
    for idx, pattern in enumerate(sorted(true_patterns), start=1):
        prob = valid_probability_distribution.get(pattern, 0.0)
        print(f"Pattern {idx:>2}: {prob:.4f}")


def total_variation(G, target_patterns, z_dim, device, num_samples, threshold=0.5):
    """
    Compute the total variation distance between the generated empirical
    distribution and the uniform target distribution over valid Bars and Stripes patterns.

    :param G: Generator model
    :param target_patterns: Tensor containing valid Bars and Stripes patterns
    :param z_dim: Dimension of the latent noise vector
    :param device: Device used for computation
    :param num_samples: Number of generated samples used for evaluation
    :param threshold: Threshold for binarising generated samples
    """
    G.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        samples = G(z)
        samples = (samples >= threshold).float().cpu()

    target_patterns = target_patterns.cpu()
    true_patterns = set(tuple(x.tolist()) for x in target_patterns)
    counts = Counter(tuple(s.tolist()) for s in samples)

    uniform_prob = 1.0 / len(true_patterns)
    tv = 0.0

    # Mismatch between empirical and target probabilities for valid patterns
    for pattern in true_patterns:
        empirical_prob = counts.get(pattern, 0) / num_samples
        tv += abs(empirical_prob - uniform_prob)

    # Probability mass assigned to invalid patterns
    for pattern, count in counts.items():
        if pattern not in true_patterns:
            tv += count / num_samples

    tv *= 0.5

    G.train()

    print(f"\nTV distance: {tv:.4f}")