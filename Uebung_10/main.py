
import sys

import numpy as np
from matplotlib import pyplot as plt


def E(akx, E0, beta):
    return E0 - beta * (2 + np.cos(akx))


def main(argv: list) -> int:

    akx = np.linspace(-np.pi, np.pi, 10000)

    E0V = 0  # eV
    E0L = 12  # eV
    betaV = -0.8  # eV
    betaL = 1.5  # eV

    plt.figure(figsize=(10, 8))

    plt.plot(akx, E(akx, E0V, betaV), label="V")
    plt.plot(akx, E(akx, E0L, betaL), label="L")

    plt.xlabel("$ak_x$")
    plt.ylabel("E")

    plt.legend()
    plt.savefig("a.png")
    plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
