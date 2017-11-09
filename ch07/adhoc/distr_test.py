import numpy as np

from lib import common

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
    plt.bar(p, vec, width=0.5)
    plt.savefig(name + ".png")


if __name__ == "__main__":
    np.random.seed(123)
    atoms = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)

    # single peak distribution
    src_hist = np.zeros(shape=(1, N_ATOMS), dtype=np.float32)
    src_hist[0, N_ATOMS//2+1] = 1.0
    save_distr(src_hist[0], "peak-01")
    proj_hist = common.distr_projection(src_hist, np.array([2], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(proj_hist[0], "peak-02")

    # normal distribution
    data = np.random.normal(size=1000, scale=3)
    hist = np.histogram(data, bins=np.arange(Vmin - DELTA_Z/2, Vmax + DELTA_Z*3/2, DELTA_Z))
    save_distr(hist[0], "normal-01")

    src_hist = hist[0]
    proj_hist = common.distr_projection(np.array([src_hist]), np.array([2], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(proj_hist[0], "normal-02")

    # normal distribution, but done episode
    proj_hist = common.distr_projection(np.array([src_hist]), np.array([2], dtype=np.float32), np.array([True]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(proj_hist[0], "normal-03")

    # clipping for out-of-range distribution
    proj_dist = common.distr_projection(np.array([src_hist]), np.array([10], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(proj_dist[0], "normal-04")

    proj_dist = common.distr_projection(np.array([src_hist]), np.array([10], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(proj_dist[0], "normal-05")

    # test both done and not done, unclipped
    proj_hist = common.distr_projection(np.array([src_hist, src_hist]), np.array([2, 2], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(proj_hist[0], "both_not_clip-01-incomplete")
    save_distr(proj_hist[1], "both_not_clip-02-complete")

    # test both done and not done, clipped right
    proj_hist = common.distr_projection(np.array([src_hist, src_hist]), np.array([10, 10], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(proj_hist[0], "both_clip-right-01-incomplete")
    save_distr(proj_hist[1], "both_clip-right-02-complete")

    # test both done and not done, clipped left
    proj_hist = common.distr_projection(np.array([src_hist, src_hist]), np.array([-10, -10], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(proj_hist[0], "both_clip-left-01-incomplete")
    save_distr(proj_hist[1], "both_clip-left-02-complete")

    pass
