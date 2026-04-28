import matplotlib.pyplot as plt
from bars_and_stripes import *
import numpy as np


def single_sample_plot(k):
    """
    Plot a single 3x3 binary sample from a flat tensor or array.
    """
    if hasattr(k, "detach"):  # torch tensor
        k = k.detach().cpu().numpy()
    else:
        k = np.asarray(k)

    k = k.reshape(3, 3)

    plt.figure(figsize=(2,2))
    plt.imshow(k, cmap='gray', vmin=0, vmax=1)
    plt.grid(color='gray', linewidth=2)
    plt.xticks([])
    plt.yticks([])

    for i in range(3):
        for j in range(3):
            # display numerical values on the pixels
            text = plt.text(
                i,
                j,
                k[j][i],
                ha="center",
                va="center",
                color="grey",
                fontsize=12,
            )
    plt.show()


def all_patterns_plots(patterns):
    """
    Plot all valid 3x3 bars-and-stripes patterns.
    """
    plt.figure(figsize=(4,4))

    for subplot_idx, pattern in enumerate(patterns, start=1):
        sample = np.asarray(pattern).reshape(3, 3)

        plt.subplot(4, 4, subplot_idx)
        plt.imshow(sample, cmap="gray", vmin=0, vmax=1)

        # display numerical values on the pixels
        for row in range(3):
            for col in range(3):
                plt.text(
                    col,
                    row,
                    int(sample[row, col]),
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=8,
                )

        plt.grid(color="gray", linewidth=2)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def define_and_visualise_target_distributions(plotting=False, data=make_bars_and_stripes(3), n_pixels=9):
    """
    Define and optionally visualise the target distribution over all 2^n_pixels
    possible patterns (512 patterns for n_pixels = 9).
    """
    probs = np.zeros(2**n_pixels)
    bitstrings, nums = represent_as_integers(data)
    probs[nums] = 1 / len(data)

    if plotting:
        plt.figure(figsize=(20, 8))
        plt.bar(np.arange(2**n_pixels), probs, width=2.0, label=r"$\pi(x)$")
        plt.xticks(nums, bitstrings, rotation=45)

        plt.xlabel("Patterns")
        plt.ylabel("Probability Distribution")
        plt.legend(loc="upper right")
        plt.subplots_adjust(bottom=0.3)
        plt.show()

    return probs


def plot_training_results(loss, kl_div):
    """
    Plot training metrics over iterations.

    Displays MMD loss and KL divergence across training iterations (for QCBM model).
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(loss)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("MMD Loss")

    ax[1].plot(kl_div)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("KL Divergence")

    plt.show()


def mark_invalid_patterns(preds, mask, N=None, n=3):
    """
    Visualises generated patterns and highlights invalid ones in red.

    :param n: dimension of each pattern (n x n)
    """
    plt.figure(figsize=(8, 8))
    j = 1
    for i, m in zip(preds[:64], mask[:64]):
        ax = plt.subplot(8, 8, j)
        j += 1
        plt.imshow(np.reshape(i, (n, n)), cmap="gray", vmin=0, vmax=1)
        if ~m:
            plt.setp(ax.spines.values(), color="red", linewidth=3)

        plt.xticks([])
        plt.yticks([])
    if N is not None:
        plt.suptitle(f"Generated patterns ({N} shots, red = invalid)", fontsize=12)
    plt.show()


def compare_px_and_py(qcbm_probs, probs, nums, bitstrings, size=9):
    """
    Compare generated and target distributions over all 2^(size patterns).

    Overlays the learned distribution p_theta(x) and target distribution pi(x),
    defined on the full pattern space.
    """
    plt.figure(figsize=(20, 8))

    plt.bar(
        np.arange(2**size),
        probs,
        width=2.0,
        label=r"$\pi(x)$",
        alpha=0.4,
        color="tab:green",
    )

    plt.bar(
        np.arange(2**size),
        qcbm_probs,
        width=2.0,
        label=r"$p_\theta(x)$",
        alpha=0.9,
        color="tab:red",
    )

    plt.xlabel("Samples")
    plt.ylabel("Probability Distribution")
    plt.xticks(nums, bitstrings, rotation=45)
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.3)
    plt.show()


