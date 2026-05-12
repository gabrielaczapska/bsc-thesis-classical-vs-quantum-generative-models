from collections import Counter
import torch


def evaluate_generated_distribution(G, target_patterns, z_dim, device, num_samples=5000,threshold=0.5):
    """
    Evaluates how well generator G covers the target patterns by sampling from G, binarising the output and comparing
    the generated patterns against the target patterns.
    
    Reports support validity, support coverage, and the empirical probability distribution over the valid patterns.

    :param z_dim: dimensionality of latent noise vector
    :param threshold: threshold for binarising generator outputs
    :return: none, prints evaluation statistics
    """

    G.eval()

    with torch.no_grad():
        # sammpling
        z = torch.randn(num_samples, z_dim, device=device)
        samples = ((G(z) > threshold).float().cpu())

    true_patterns = [tuple(x.cpu().tolist()) for x in target_patterns]
    true_set = set(true_patterns)
    
    counts = Counter(tuple(s.tolist()) for s in samples)

    valid_counts = {k: v for k, v in counts.items() if k in true_set}
    invalid_counts = {k: v for k, v in counts.items() if k not in true_set}

    total_valid = sum(valid_counts.values())
    total_invalid = sum(invalid_counts.values())

    # metrics
    valid_ratio = total_valid / num_samples
    invalid_ratio = total_invalid / num_samples
    coverage_ratio = len(valid_counts) / len(true_set)

    valid_probability_distribution = {
    k: v / total_valid if total_valid > 0 else 0.0
    for k, v in valid_counts.items()
    }

    G.train()
    
    print(f"\nNumber of generated samples: {num_samples} ")
    print("\nSupport Validity")
    print(f"Valid samples :         {total_valid} ({valid_ratio:.4f})")
    print(f"Invalid samples :       {total_invalid} ({invalid_ratio:.4f})")

    print(f"Support Coverage")
    print(f"Covered valid patterns: {len(valid_counts)}/{len(true_set)}  ({coverage_ratio:.4f})")

    print("\nEmpirical Distribution over Valid Patterns")
    for idx, pattern in enumerate(sorted(true_set), start=1):
        prob = valid_probability_distribution.get(pattern, 0.0)
        print(f"Pattern {idx:>2}: {prob:.4f}")


def total_variation(G, target_patterns, z_dim, device, num_samples=5000, threshold=0.5):
    """
    Computes total variation distance between the empirical distribution over valid patterns
    and the uniform target distribution.
    """
    G.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        samples = (G(z) > threshold).float().cpu()

    true_patterns = [tuple(x.cpu().tolist()) for x in target_patterns]
    true_set = set(true_patterns)

    counts = Counter(tuple(s.tolist()) for s in samples)
    valid_counts = {k: v for k, v in counts.items() if k in true_set}

    empirical = torch.tensor(
        [counts.get(p, 0) / num_samples for p in true_patterns],
        dtype=torch.float32,
    )
    uniform = torch.full_like(empirical, 1.0 / len(true_patterns))

    tv = 0.5 * torch.abs(empirical - uniform).sum().item()

    G.train()

    print(f"\nTV distance: {tv:.4f}")
