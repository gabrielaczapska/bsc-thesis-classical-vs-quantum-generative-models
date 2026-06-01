"""
Evaluation metrics for generative models trained on the 3x3 Bars and Stripes dataset.

    Supports:
    - WGAN-GP generators
    - QCBM circuit
"""

from collections import Counter
import numpy as np
import torch
import pennylane as qml


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

    true_patterns = set(tuple(p.tolist()) for p in target_patterns)
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


def evaluate_qcbm(weights, target_patterns, sampling_circuit, num_samples=5000):
    """
    Evaluate the QCBM using the same evaluation metrics as the WGAN-GP models.
    Compare the empirical distribution to the target distribution.

    Report support validity, support coverage, the empirical distribution over valid patterns
    and total variation distance.

    :param weights: Trainable QCBM circuit parameters
    :param target_patterns: Tensor containing valid Bars and Stripes patterns
    :param sampling_circuit: Quantum circuit used to generate samples
    :param num_samples: Number of generated samples used for evaluation
    """
    circ = qml.set_shots(sampling_circuit, shots=num_samples)

    samples = np.asarray(circ(weights), dtype=int)

    target_patterns = target_patterns.cpu()

    true_patterns = [
        tuple(np.asarray(pattern, dtype=int).flatten().tolist())
        for pattern in target_patterns
    ]

    true_set = set(true_patterns)

    counts = Counter(tuple(sample.flatten.tolist()) for sample in samples)

    valid_counts = {k: v for k, v in counts.items() if k in true_set}
    invalid_counts = {k: v for k, v in counts.items() if k not in true_set}

    total_valid = sum(valid_counts.values())
    total_invalid = sum(invalid_counts.values())

    # Empirical probability distribution over valid samples only
    valid_probability_distribution = {
        pattern: count / total_valid if total_valid > 0 else 0.0
        for pattern, count in valid_counts.items()
    }

    # Empirical probability distribution over all generated samples
    empirical = torch.tensor(
        [counts.get(pattern, 0) / num_samples for pattern in true_patterns],
        dtype=torch.float32
    )

    uniform = torch.full_like(empirical, 1.0 / len(true_patterns))

    # Evaluation metrics
    validity_ratio = total_valid / num_samples
    invalidity_ratio = total_invalid / num_samples
    coverage_ratio = len(valid_counts) / len(true_patterns)


    # Total variation distance over the valid support
    tv = 0.5 * torch.abs(empirical - uniform).sum().item()

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

    print(f"\nTV distance: {tv:.4f}")
