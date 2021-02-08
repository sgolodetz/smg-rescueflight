import numpy as np

from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter


def main():
    camera: SimpleCamera = SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0])
    pose: np.ndarray = CameraPoseConverter.camera_to_pose(camera)
    print(pose)

    model_view: np.ndarray = CameraPoseConverter.pose_to_modelview(pose)
    print(model_view)


if __name__ == "__main__":
    main()
