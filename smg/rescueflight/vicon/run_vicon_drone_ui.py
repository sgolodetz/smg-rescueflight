from argparse import ArgumentParser
from typing import Tuple

from smg.meshing import MeshUtil
from smg.rescueflight.vicon.vicon_drone_ui import ViconDroneUI
from smg.rotory.drones import Tello


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_controller_type", "-t", type=str, default="keyboard", choices=("futaba_t6k", "keyboard", "rts"),
        help="the type of drone controller to use"
    )
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

    # Construct the drone.
    # TODO

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Construct the drone.
    with Tello(print_commands=True, print_responses=True) as drone:
        # Construct the Vicon drone UI.
        with ViconDroneUI(
            debug=False,
            drone=drone,
            drone_controller_type=args.get("drone_controller_type"),
            drone_mesh=MeshUtil.load_tello_mesh(),
            intrinsics=intrinsics,
            planning_octree_filename=args.get("planning_octree"),
            scene_mesh_filename=args.get("scene_mesh"),
            scene_octree_filename=args.get("scene_octree")
        ) as drone_ui:
            # Run the Vicon drone UI.
            drone_ui.run()


if __name__ == "__main__":
    main()
