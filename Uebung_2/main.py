import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def Y_s():
    return np.sqrt(1 / (4 * np.pi))


def Y_px(theta, phi):
    return np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.cos(phi)


def Y_py(theta, phi):
    return np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)


def Y_pz(theta):
    return np.sqrt(3 / (4 * np.pi)) * np.cos(theta)


def psi_1_sp1(theta, phi):
    return (1 / (np.sqrt(2))) * (Y_s() + Y_px(theta, phi))


def psi_2_sp1(theta, phi):
    return (1 / (np.sqrt(2))) * (Y_s() - Y_px(theta, phi))


def psi_1_sp2(theta, phi):
    return ((1 / np.sqrt(3)) * Y_s()) + ((1 / np.sqrt(6)) * Y_px(theta, phi)) + ((1 / np.sqrt(2)) * Y_py(theta, phi))


def psi_2_sp2(theta, phi):
    return ((1 / np.sqrt(3)) * Y_s()) + ((1 / np.sqrt(6)) * Y_px(theta, phi)) - ((1 / np.sqrt(2)) * Y_py(theta, phi))


def psi_1_sp3(theta, phi):
    return 0.5 * (Y_s() + Y_py(theta, phi) + Y_px(theta, phi) + Y_pz(theta))


def psi_2_sp3(theta, phi):
    return 0.5 * (Y_s() - Y_py(theta, phi) + Y_px(theta, phi) + Y_pz(theta))


def psi_3_sp3(theta, phi):
    return 0.5 * (Y_s() + Y_py(theta, phi) - Y_px(theta, phi) - Y_pz(theta))


def psi_4_sp3(theta, phi):
    return 0.5 * (Y_s() - Y_py(theta, phi) - Y_px(theta, phi) + Y_pz(theta))


def abs_sq(x):
    return abs(x * x)


def spherical_to_cartesian(r, theta, phi):
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)


def normalize(x: np.ndarray):
    return np.interp(x, (x.min(), x.max()), (0, 1))
    # return x / np.linalg.norm(x)


def u_to_kg(m):
    return 1.66053904e-27 * m


def J_to_eV(E):
    return E / 1.602176634e-19


def aufgabe_1(plot=True):
    use_radius = True

    theta, phi = np.mgrid[0: np.pi: 80j, 0: 2 * np.pi: 160j]

    r = 1

    x, y, z = spherical_to_cartesian(r, theta, phi)

    functions = [psi_1_sp1, psi_2_sp1, psi_1_sp2, psi_2_sp2, psi_1_sp3, psi_2_sp3, psi_3_sp3, psi_4_sp3]

    for f in functions:
        c = abs_sq(f(theta, phi))

        if use_radius:
            r = c
            x, y, z = spherical_to_cartesian(r, theta, phi)

        if plot:
            fig = plt.figure(figsize=(10, 8))

            ax = fig.gca(projection="3d")

            color = cm.viridis(normalize(c))

            cax = ax.plot_surface(x, y, z, facecolors=color, antialiased=False, cmap="viridis", vmin=c.min(), vmax=c.max())

            cbar = fig.colorbar(cax)

            cbar.ax.set_yticklabels(np.round(np.linspace(c.min(), c.max(), 9), 3))

            plt.title(f.__name__)
            plt.xlabel("x")
            plt.ylabel("y")

            plt.show()


def aufgabe_2():
    def energy(v, l, nu, I):
        return (v + 0.5) * consts.hbar * 2 * consts.pi * nu + l * (l + 1) * consts.hbar ** 2 / (2 * I)

    mH = u_to_kg(1.008)  # kg
    mCl = u_to_kg(35.45)  # kg

    m = mH * mCl / (mH + mCl)  # kg
    R = 127.5e-12  # m

    I = m * R ** 2  # kg m^2

    l1s = [1, 2, 3, 2]
    l2s = [0, 1, 4, 3]

    v1 = 0
    v2 = 1

    nu = consts.c * 2988.9e2  # s^-1

    for l1, l2 in zip(l1s, l2s):
        delta_E = abs(energy(v1, l1, nu, I) - energy(v2, l2, nu, I))
        print("delta_E =", delta_E, "J =", J_to_eV(delta_E), "eV =", delta_E * 1e-12 / consts.h, "THz")


def main(argv: list) -> int:
    aufgabe_1(plot=True)
    aufgabe_2()

    return 0


if __name__ == "__main__":
    main(sys.argv)
