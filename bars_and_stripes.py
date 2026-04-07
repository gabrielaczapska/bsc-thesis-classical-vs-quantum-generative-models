import numpy as np

def get_bars_and_stripes(n):
    bitstrings = [list(np.binary_repr(i,n))[::-1] for i in range(2**n)]
    bitstrings = np.array(bitstrings, dtype=int)

    # let n = 3
    # bitstrings - array with 8 entries

    stripes = bitstrings.copy()
    # creates an object
    stripes = np.repeat(stripes, n, 0)
    stripes = stripes.reshape(2**n, n*n)

    bars = bitstrings.copy()
    bars = bars.reshape(2**n * n, 1)
    bars = np.repeat(bars, n, 1)
    bars = bars.reshape(2**n, n * n)
    #return stripes.shape, bars.shape
    return np.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))

data = get_bars_and_stripes(3)