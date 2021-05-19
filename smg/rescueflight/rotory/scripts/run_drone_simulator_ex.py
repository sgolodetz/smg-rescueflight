from typing import Tuple

from drone_simulator import DroneSimulator


def main() -> None:
    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Construct the drone simulator.
    with DroneSimulator(
        intrinsics=intrinsics, plan_paths=True, tello_mesh_filename="C:/smglib/meshes/tello.ply"
    ) as simulator:
        # Run the simulator.
        simulator.run()


if __name__ == "__main__":
    main()
