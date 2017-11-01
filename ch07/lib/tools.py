import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# Vmax = 10
# Vmin = -10
# N_ATOMS = 51
# DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


def save_distr(atoms, vec, name):
    plt.cla()
#    p = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)
    plt.bar(atoms, vec, width=0.5)
    plt.savefig(name + ".png")


def distr_projection(next_distr, rewards, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = rewards + (Vmin + atom * delta_z) * gamma
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j)
        u = np.ceil(b_j)
        proj_distr[:, atom] += next_distr[:, atom] * (u - b_j)
        proj_distr[:, atom] += next_distr[:, atom] * (b_j - l)
    return proj_distr
