import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import CubicSpline
from timeit import default_timer as timer
from typing import List


def main() -> None:
    y: np.ndarray = np.array([
        [0.025, 0.025, 0.025],
        [0.075, 0.025, 0.075],
        [0.125, 0.025, 0.125],
        [0.175, 0.025, 0.125],
        [0.225, 0.025, 0.125],
        [0.275, 0.025, 0.175],
    ])

    x: List[int] = np.arange(len(y))

    cs: CubicSpline = CubicSpline(x, y, bc_type='clamped')
    start = timer()
    iy: np.ndarray = cs(np.linspace(0, len(y) - 1, 100))
    end = timer()
    print(f"Interpolation Time: {end - start}s")

    fig, ax = plt.subplots(1, 1)
    ax.axes.set_aspect('equal')
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.2)
    ax.xaxis.set_ticks(np.arange(0, 0.35, 0.05))
    ax.yaxis.set_ticks(np.arange(0, 0.25, 0.05))
    ax.set_axisbelow(True)
    ax.grid(linestyle='-', linewidth='0.5', color='blue')
    ax.plot(y[:, 0], y[:, 2], 'o', label='data')
    ax.plot(iy[:, 0], iy[:, 2], label='spline')
    plt.show()


if __name__ == "__main__":
    main()
