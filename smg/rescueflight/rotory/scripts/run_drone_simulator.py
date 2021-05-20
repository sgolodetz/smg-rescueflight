from argparse import ArgumentParser
from typing import Tuple

from drone_simulator import DroneSimulator


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--planning_octree", type=str,
        help="the name of the planning octree file"
    )
    parser.add_argument(
        "--scene_mesh", type=str,
        help="the name of the scene mesh file"
    )
    parser.add_argument(
        "--scene_octree", type=str,
        help="the name of the scene octree file"
    )
    args: dict = vars(parser.parse_args())

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Construct the drone simulator.
    with DroneSimulator(
        drone_mesh_filename="C:/smglib/meshes/tello.ply",
        intrinsics=intrinsics,
        plan_paths=True,
        planning_octree_filename=args.get("planning_octree"),
        scene_mesh_filename=args.get("scene_mesh"),
        scene_octree_filename=args.get("scene_octree")
    ) as simulator:
        # Run the simulator.
        simulator.run()


if __name__ == "__main__":
    main()
