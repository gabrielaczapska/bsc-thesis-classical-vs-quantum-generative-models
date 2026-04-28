from collections import Counter
import torch


def chi_metric_to_uniform(counts, true_patterns, num_samples, eps=1e-8):
    """
    Chi-square style distance between the empirical generated distribution
    and a uniform distribution over the valid target patterns.
    """
    expected_prob = 1.0 / len(true_patterns)

    chi = 0.0
    for pattern in true_patterns:
        observed_prob = counts.get(pattern, 0) / num_samples
        chi += ((observed_prob - expected_prob) ** 2) / (expected_prob + eps)

    return chi


def kl_divergence_to_uniform(valid_counts, true_patterns, eps=1e-8):
    """
    KL divergence over VALID SUPPORT only.
    """
    total_valid = sum(valid_counts.values())

    if total_valid == 0:
        return float("inf")  # or handle edge case

    num_patterns = len(true_patterns)
    q = 1.0 / num_patterns

    kl = 0.0
    for pattern in true_patterns:
        p = valid_counts.get(pattern, 0) / total_valid

        if p > 0:
            kl += p * torch.log(torch.tensor(p / (q + eps))).item()

    return kl


def support_coverage_stats(G, target_patterns, z_dim, device, num_samples=5000,threshold=0.5, chunk_size=None,
                           top_k_invalid=10, show_flat_patterns=False, compute_chi_metric=True):
    """
    Purpose:
    Evaluates how well generator G covers the target patterns by sampling from G, binarising the output and comparing
    the generated patterns against the target patterns.
    Reports coverage, frequency statistics, and most frequent invalid patterns.

    :param z_dim: dimensionality of latent input
    :param threshold: threshold for binarising generator outputs
    :param top_k_invalid: number of most frequent invalid patterns to display
    :return: none, prints coverage statistics and pattern frequencies
    """

    G.eval()

    with torch.no_grad():
        if chunk_size is None or chunk_size >= num_samples:
            z = torch.randn(num_samples, z_dim, device=device)
            samples = (G(z) > threshold).float().cpu()
        else:
            samples_all = []
            for start in range(0, num_samples, chunk_size):
                cur = min(chunk_size, num_samples - start)
                z = torch.randn(cur, z_dim, device=device)
                samples_all.append((G(z) > threshold).float().cpu())
            samples = torch.cat(samples_all, dim=0)

    true_patterns = [tuple(x.cpu().tolist()) for x in target_patterns]
    true_set = set(true_patterns)
    counts = Counter(tuple(s.tolist()) for s in samples)

    valid_counts = {k: v for k, v in counts.items() if k in true_set}
    invalid_counts = {k: v for k, v in counts.items() if k not in true_set}

    total_valid = sum(valid_counts.values())
    total_invalid = sum(invalid_counts.values())

    valid_ratio = total_valid / num_samples
    invalid_ratio = total_invalid / num_samples

    print(f"\nGenerated {num_samples} samples:")
    print(f"Covered valid patterns: {len(valid_counts)}/{len(true_set)}")
    print(f"Invalid unique patterns: {len(invalid_counts)}")

    print("\nOverall Quality:")
    print(f"Total valid samples:   {total_valid} ({valid_ratio:.4f})")
    print(f"Total invalid samples: {total_invalid} ({invalid_ratio:.4f})")

    if compute_chi_metric:
        chi_metric = chi_metric_to_uniform(counts, true_patterns, num_samples)
        print(f"Chi metric to uniform valid distribution: {chi_metric:.6f}")

    if show_flat_patterns:
        print("\nValid Pattern Frequencies:")
        for i, pattern in enumerate(sorted(true_set), start=1):
            c = valid_counts.get(pattern, 0)
            p = c / num_samples
            print(f"\nPattern {i}: count={c}, prob={p:.4f}")
            print(torch.tensor(pattern))  # ← flattened

        if invalid_counts:
            print("\nInvalid Patterns (top by frequency):")
            for i, (pattern, c) in enumerate(
                    sorted(invalid_counts.items(), key=lambda x: -x[1])[:top_k_invalid],
                    start=1,
            ):
                print(f"\nInvalid {i}: count={c}, prob={c / num_samples:.4f}")
                print(torch.tensor(pattern))  # ← flattened

    G.train()


def total_variation_to_uniform(G, target_patterns, z_dim, device, num_samples=5000, threshold=0.5, compute_kl_metric=True):

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

    print(f"\nTV distance to uniform over {len(true_patterns)} valid patterns: {tv:.4f}")
    print("\nEmpirical probs over valid support:")
    for i, p in enumerate(empirical.tolist(), start=1):
        print(f"pattern {i}: {p:.4f}")

    if compute_kl_metric:
        kl_metric = kl_divergence_to_uniform(valid_counts, true_patterns)
        print(f"\nKL divergence over valid support: {kl_metric:.6f}")

    G.train()