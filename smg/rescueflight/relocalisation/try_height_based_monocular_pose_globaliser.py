import matplotlib.pyplot as plt
import numpy as np

from smg.relocalisation.poseglobalisers import HeightBasedMonocularPoseGlobaliser


def main() -> None:
    globaliser: HeightBasedMonocularPoseGlobaliser = HeightBasedMonocularPoseGlobaliser(debug=True)

    for i in range(50):
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
    globaliser.finish_training()


if __name__ == "__main__":
    main()
