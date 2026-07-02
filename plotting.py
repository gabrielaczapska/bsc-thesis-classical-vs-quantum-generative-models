"""
Visualisation of a single 3x3 Bars and Stripes pattern, all target patterns, and the target distribution.
"""

import matplotlib.pyplot as plt
from bars_and_stripes import make_bars_and_stripes, represent_as_integers
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


def define_and_visualise_target_distributions(data=make_bars_and_stripes(3), n_pixels=9):
    """
    Define and optionally visualise the target distribution over all 2^n_pixels
    possible patterns (512 patterns for n_pixels = 9).
    """
    # define the distribution
    probs = np.zeros(2**n_pixels)
    bitstrings, nums = represent_as_integers(data)
    probs[nums] = 1 / len(data)

    # plot the distribution
    plt.figure(figsize=(20, 8))
    plt.bar(np.arange(2**n_pixels), probs, width=2.0, label=r"$\pi(x)$")
    plt.xticks(nums, bitstrings, rotation=45)

    plt.xlabel("Patterns")
    plt.ylabel("Probability Distribution")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.3)
    plt.show()

    return probs
