import torch


def make_bars_and_stripes(n=3):
    """
    Generate all unique n×n Bars and Stripes patterns.

    For n=3, this produces 14 patterns.

    :param n: The size of the pattern grid, where each pattern has a shape of nxn
    :return: torch.Tensor of shape (num_patterns, n*n) containing all valid patterns flattened to vectors
    """
    patterns = []

    # Stripes: each row is all 0s or all 1s
    for mask in range(2 ** n):
        x = torch.zeros(n, n)
        for i in range(n):
            bit = (mask >> i) & 1
            x[i, :] = bit
        patterns.append(x)

    # Bars: each column is all 0s or all 1s
    for mask in range(2 ** n):
        x = torch.zeros(n, n)
        for j in range(n):
            bit = (mask >> j) & 1
            x[:, j] = bit
        patterns.append(x)

    # Remove duplicates
    unique = []
    seen = set()
    for x in patterns:
        key = tuple(x.flatten().tolist())
        if key not in seen:
            seen.add(key)
            unique.append(x)

    data = torch.stack(unique)
    data = data.view(-1, n * n).float()

    return data


def sample_real_batch(data, batch_size):
    """
    Randomly sample a mini-batch from the real dataset.

    :param data: Tensor containing real data samples
    :param batch_size: Number of samples to draw
    :return: Tensor containing the sampled batch
    """
    idx = torch.randint(0, data.size(0), (batch_size,))
    return data[idx]


def represent_as_integers(data=make_bars_and_stripes(3)):
    """
    Produce the integer representation of each valid Bars and Stripes pattern.

    :param data: Tensor containing valid Bars and Stripes patterns
    :return: Tuple containing:
             - list of binary string representations
             - list of corresponding integer values
    """
    bitstrings = []
    nums = []

    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
    # nums: [0, 292, 146, 438, 73, 365, 219, 448, 56, 504, 7, 455, 63, 511]
    return bitstrings, nums
