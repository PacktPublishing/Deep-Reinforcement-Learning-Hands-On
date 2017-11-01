import numpy as np

from lib import tools

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


if __name__ == "__main__":
    np.random.seed(123)
    atoms = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)

    # single peak distribution
    src_hist = np.zeros(shape=(1, N_ATOMS), dtype=np.float32)
    src_hist[0, N_ATOMS//2+1] = 1.0
    tools.save_distr(atoms, src_hist[0], "peak-01")
    proj_hist = tools.distr_projection(src_hist, np.array([2], dtype=np.float32),
                                       Vmin, Vmax, N_ATOMS, gamma=0.9)
    tools.save_distr(atoms, proj_hist[0], "peak-02")

    # normal distribution
    data = np.random.normal(size=1000, scale=3)
    hist = np.histogram(data, bins=np.arange(Vmin - DELTA_Z/2, Vmax + DELTA_Z*3/2, DELTA_Z))
    tools.save_distr(atoms, hist[0], "normal-01")

    src_hist = hist[0]
    proj_hist = tools.distr_projection(np.array([src_hist]), np.array([2], dtype=np.float32),
                                       Vmin, Vmax, N_ATOMS, gamma=0.9)
    tools.save_distr(atoms, proj_hist[0], "normal-02")

    pass
