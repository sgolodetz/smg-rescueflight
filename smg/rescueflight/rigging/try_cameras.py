import math
import numpy as np

from smg.rigging.cameras import CompositeCamera, DerivedCamera


def print_composite_camera(composite: CompositeCamera):
    print(f"Composite: p={composite.p()}, n={composite.n()}, u={composite.u()}, v={composite.v()}")
    for name, cam in composite.get_secondary_cameras().items():
        print(f"  - {name}: p={cam.p()}, n={cam.n()}, u={cam.u()}, v={cam.v()}")


def main():
    np.set_printoptions(suppress=True)

    up: np.ndarray = np.array([0, -1, 0])
    composite_cam: CompositeCamera = CompositeCamera([0, 0, 0], [0, 0, 1], up)
    composite_cam.add_secondary_camera("Left Eye", DerivedCamera(composite_cam, np.eye(3), [1, 0, 0]))
    composite_cam.add_secondary_camera("Right Eye", DerivedCamera(composite_cam, np.eye(3), [-1, 0, 0]))
    print_composite_camera(composite_cam)

    composite_cam.move_n(1)
    print_composite_camera(composite_cam)

    composite_cam.rotate(up, math.pi / 2)
    print_composite_camera(composite_cam)


if __name__ == "__main__":
    main()
