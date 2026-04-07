import matplotlib.pyplot as plt
from bars_and_stripes import *


# plot of a single sample i
def single_sample_plot(i):

    plt.figure(figsize=(2,2))
    plt.imshow(i, cmap='gray', vmin=0, vmax=1)
    plt.grid(color='gray', linewidth=2)
    plt.xticks([])
    plt.yticks([])

    for i in range(3):
        for j in range(3):
            # show the numerical values on the pixels
            text = plt.text(i,j, sample[j,i], ha="center", va="center", color="grey", fontsize=12)

    plt.show()
    print(f"\nSample bitstring: {''.join(np.array(sample.flatten(), dtype='str'))}")


# plot of all samples
def all_patterns_plots(patterns):
    plt.figure(figsize=(4,4))
    subplot_idx = 1
    for i in patterns:
        ax = plt.subplot(4, 4, subplot_idx)
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


# single sample call
data = get_bars_and_stripes(3)
sample = data[3].reshape(3,3)
single_sample_plot(sample)


# all patterns visualisation
all_patterns_plots(data)