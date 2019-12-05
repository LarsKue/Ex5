
import sys
import numpy as np
from matplotlib import pyplot as plt


def w(q_i, w0, a):
    return 2 * w0 * np.abs(np.sin(q_i * a / 2))


def main(argv: list) -> int:

    a = 3e-10  # m
    v = 2000  # m/s
    w0 = v / a

    q_i = np.linspace(-np.pi / 2, np.pi / 2, 4000)
    plt.figure(figsize=(8, 8))
    plt.plot(q_i, w(q_i, w0, a))
    plt.xlabel(r"$q_i$")
    plt.ylabel(r"$\omega$")
    plt.show()



    return 0


if __name__ == "__main__":
    main(sys.argv)