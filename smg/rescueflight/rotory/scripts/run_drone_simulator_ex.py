from typing import Tuple

from drone_simulator import DroneSimulator


def main() -> None:
    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Construct the drone simulator.
    with DroneSimulator(
        drone_mesh_filename="C:/smglib/meshes/tello.ply",
        intrinsics=intrinsics,
        plan_paths=True,
        scene_mesh_filename="C:/smglib/smg-mapping/output-navigation/mesh.ply",
        scene_octree_filename="C:/smglib/smg-mapping/output-navigation/octree5cm.bt"
    ) as simulator:
        # Run the simulator.
        simulator.run()


if __name__ == "__main__":
    main()
