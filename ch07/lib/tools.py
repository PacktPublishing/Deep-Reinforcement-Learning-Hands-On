import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


def save_distr(vec, name):
    plt.cla()
    p = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)
    plt.bar(p, vec)
    plt.savefig(name + ".png")
