import sys
import numpy as np
sys.path.append("./")

from lib import common

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


def save_distr(src, proj, name):
    plt.clf()
    p = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)
    plt.subplot(2, 1, 1)
    plt.bar(p, src, width=0.5)
    plt.title("Source")
    plt.subplot(2, 1, 2)
    plt.bar(p, proj, width=0.5)
    plt.title("Projected")
    plt.savefig(name + ".png")


if __name__ == "__main__":
    np.random.seed(123)
    atoms = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)

    # single peak distribution
    src_hist = np.zeros(shape=(1, N_ATOMS), dtype=np.float32)
    src_hist[0, N_ATOMS//2+1] = 1.0
    proj_hist = common.distr_projection(src_hist, np.array([2], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist[0], proj_hist[0], "peak-r=2")

    # normal distribution
    data = np.random.normal(size=1000, scale=3)
    hist = np.histogram(data, normed=True, bins=np.arange(Vmin - DELTA_Z/2, Vmax + DELTA_Z*3/2, DELTA_Z))

    src_hist = hist[0]
    proj_hist = common.distr_projection(np.array([src_hist]), np.array([2], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_hist[0], "normal-r=2")

    # normal distribution, but done episode
    proj_hist = common.distr_projection(np.array([src_hist]), np.array([2], dtype=np.float32), np.array([True]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_hist[0], "normal-done-r=2")

    # clipping for out-of-range distribution
    proj_dist = common.distr_projection(np.array([src_hist]), np.array([10], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_dist[0], "normal-r=10")

    # test both done and not done, unclipped
    proj_hist = common.distr_projection(np.array([src_hist, src_hist]), np.array([2, 2], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_not_clip-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_not_clip-02-complete")

    # test both done and not done, clipped right
    proj_hist = common.distr_projection(np.array([src_hist, src_hist]), np.array([10, 10], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_clip-right-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_clip-right-02-complete")

    # test both done and not done, clipped left
    proj_hist = common.distr_projection(np.array([src_hist, src_hist]), np.array([-10, -10], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_clip-left-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_clip-left-02-complete")

    pass
