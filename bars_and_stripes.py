import torch


def make_bars_and_stripes(n=3):
    """
    Generates all unique n×n bars-and-stripes patterns.

    :param n: size of the grid, each pattern has shape (n, n)
    :return: torch.Tensor: tensor of shape (num_patterns, n*n) containing all valid patterns flattened to vectors
    """
    patterns = []

    # stripes (rows)
    for mask in range(2 ** n):
        x = torch.zeros(n, n)
        for i in range(n):
            bit = (mask >> i) & 1
            x[i, :] = bit
        patterns.append(x)

    # bars (columns)
    for mask in range(2 ** n):
        x = torch.zeros(n, n)
        for j in range(n):
            bit = (mask >> j) & 1
            x[:, j] = bit
        patterns.append(x)

    # remove duplicates
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
    Samples a mini-batch of the real data.

    :return: batch of samples
    """
    idx = torch.randint(0, data.size(0), (batch_size,))
    return data[idx]


def is_bars_and_stripes(x):
    """
    Checks whether the pattern is a valid bars-and-stripes configuration

    :return: checks if x is one of the valid bars and stripes patterns
    """
    if x.dim() == 1:
        x = x.view(3, 3)

    rows_ok = all(torch.all(x[i, :] == x[i, 0]) for i in range(3))
    cols_ok = all(torch.all(x[:, j] == x[0, j]) for j in range(3))

    return rows_ok or cols_ok


def represent_as_integers(data=make_bars_and_stripes(3)):
    """
    :param data: set of bars and stripes patterns
    :return: integer values corresponding to the bars and stripes patterns
    """
    bitstrings = []
    nums = []

    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
    # nums: [0, 292, 146, 438, 73, 365, 219, 448, 56, 504, 7, 455, 63, 511]
    return bitstrings, nums
