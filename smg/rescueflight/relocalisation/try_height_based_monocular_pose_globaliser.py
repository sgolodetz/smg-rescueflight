import matplotlib.pyplot as plt
import numpy as np

from smg.relocalisation.poseglobalisers import HeightBasedMonocularPoseGlobaliser


def main() -> None:
    globaliser: HeightBasedMonocularPoseGlobaliser = HeightBasedMonocularPoseGlobaliser(debug=True)
    # _, ax = plt.subplots(3, 1)
    #
    # print(type(ax))
    #
    # xs = []
    # ys = []
    #
    # for i in range(100):
    #     for j in range(3):
    #         ax[j].clear()
    #
    #     xs.append(i)
    #     ys.append(i ** 2)
    #
    #     ax[0].plot(xs, ys)
    #
    #     plt.draw()
    #     plt.waitforbuttonpress(0.001)

    for i in range(20):
        tracker_i_t_c: np.ndarray = np.eye(4)
        height: float = 0.0
        k: int = i % 4
        if k == 0:
            height = 0.0
        elif k == 1:
            height = i / 4
        elif k == 2:
            height = 0.0
        elif k == 3:
            height = -i / 4
        tracker_i_t_c[1, 3] = height * 2
        globaliser.train(tracker_i_t_c, height)

    plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
