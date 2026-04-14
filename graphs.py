import matplotlib.pyplot as plt
from bars_and_stripes import *


# plot a single k-th sample
def single_sample_plot(k):
    plt.figure(figsize=(2,2))
    plt.imshow(k, cmap='gray', vmin=0, vmax=1)
    plt.grid(color='gray', linewidth=2)
    plt.xticks([])
    plt.yticks([])

    for i in range(3):
        for j in range(3):
            # show the numerical values on the pixels
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
    print(f"\nSample bitstring: {''.join(np.array(k.flatten(), dtype='str'))}")


# plot full dataset
def all_patterns_plots(patterns):
    plt.figure(figsize=(4,4))
    subplot_idx = 1
    for i in patterns:
        plt.subplot(4, 4, subplot_idx)
        sample = np.reshape(i, (3, 3))
        plt.imshow(np.reshape(i,(3, 3)), cmap='gray', vmin=0, vmax=1)
        # show the numerical values on the pixels
        for row in range(3):
            for col in range(3):
                plt.text(
                    col,
                    row,
                    sample[row, col],
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=8,
                )
        plt.grid(color='gray', linewidth=2)
        plt.xticks([])
        plt.yticks([])
        subplot_idx += 1

    plt.show()


# visualise the distribution of all target patterns
def define_and_visualise_target_distributions(plotting=False, data=get_bars_and_stripes(3), n_pixels=9):
    # assign probabilities to each of 512 patterns that can be defined on 9 pixels
    probs = np.zeros(2**n_pixels)
    bitstrings, nums = represent_as_integers(data)
    probs[nums] = 1 / len(data)

    if plotting:
        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(2**n_pixels), probs, width=2.0, label=r"$\pi(x)$")
        plt.xticks(nums, bitstrings, rotation=45)

        plt.xlabel("Patterns")
        plt.ylabel("Probability Distribution")
        plt.legend(loc="upper right")
        plt.subplots_adjust(bottom=0.3)
        plt.show()

    return probs


# plot the training results (QCBM)
def plot_training_results(loss, kl_div):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(loss)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("MMD Loss")

    ax[1].plot(kl_div)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("KL Divergence")

    plt.show()

# single sample call
#data = get_bars_and_stripes(3)
#sample = data[3].reshape(3,3)
#single_sample_plot(sample)

# all patterns visualisation
#all_patterns_plots(data)

# visualisation of probability distribution for all target patterns
#probs = define_and_visualise_target_distributions(plotting=False)
