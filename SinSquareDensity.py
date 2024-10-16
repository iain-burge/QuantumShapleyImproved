
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange



def main():
    r = 1<<20
    a = 1

    expected = []
    results  = []
    for m in trange(1,r):
        M = m#1<<m

        results.append(0)
        expected.append(np.sqrt(m)/np.pi)
        for k in range(M):
            if k <= a/M: expected[-1] += 1

            if np.sin(np.pi*k/M)**2 <= a/(2*M): results[-1] -= 1
            if np.sin(np.pi*k/M)**2 <= a/M: results[-1] += 1
            else: break

        # expected[-1] /= M
        # results[-1] /= M

    plt.plot(expected, label='expected')
    plt.plot(results,  label='results')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
