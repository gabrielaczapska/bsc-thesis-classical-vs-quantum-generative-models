import numpy as np

def get_bars_and_stripes(n):
    bitstrings = [list(np.binary_repr(i,n))[::-1] for i in range(2**n)]
    bitstrings = np.array(bitstrings, dtype=int)

    stripes = bitstrings.copy()
    stripes = np.repeat(stripes, n, 0)
    stripes = stripes.reshape(2**n, n*n)

    bars = bitstrings.copy()
    bars = bars.reshape(2**n * n, 1)
    bars = np.repeat(bars, n, 1)
    bars = bars.reshape(2**n, n * n)
    return np.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))


# output integer values corresponding to the binary notations of patterns
def represent_as_integers(data=get_bars_and_stripes(3)):
    bitstrings = []
    nums = []

    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
    # output for nums: [0, 292, 146, 438, 73, 365, 219, 448, 56, 504, 7, 455, 63, 511]
    return bitstrings, nums


#data = get_bars_and_stripes(3)
#rep, nums = represent_as_integers(data)
