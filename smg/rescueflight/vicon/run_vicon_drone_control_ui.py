from argparse import ArgumentParser
from typing import Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.meshing import MeshUtil
from smg.rescueflight.vicon.vicon_drone_control_ui import ViconDroneControlUI
from smg.rotory.drones import Tello


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_controller_type", "-t", type=str, default="keyboard",
        choices=("aws_transcribe", "futaba_t6k", "keyboard", "rts"),
        help="the type of drone controller to use"
    )
    parser.add_argument(
        "--planning_octree", type=str,
        help="the name of the planning octree file"
    )
    parser.add_argument(
        "--run_client", action="store_true",
        help="whether to stream frames to a mapping server"
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

    # Construct a mapping client if requested.
    mapping_client: Optional[MappingClient] = None
    try:
        if args["run_client"]:
            mapping_client = MappingClient(frame_compressor=RGBDFrameMessageUtil.compress_frame_message)

        # Construct the drone.
        with Tello(print_commands=False, print_responses=False, print_state_messages=False) as drone:
            # Construct the drone control UI.
            with ViconDroneControlUI(
                debug=False,
                drone=drone,
                drone_controller_type=args.get("drone_controller_type"),
                drone_mesh=MeshUtil.load_tello_mesh(),
                intrinsics=intrinsics,
                mapping_client=mapping_client,
                planning_octree_filename=args.get("planning_octree"),
                scene_mesh_filename=args.get("scene_mesh"),
                scene_octree_filename=args.get("scene_octree")
            ) as drone_control_ui:
                # Run the drone control UI.
                drone_control_ui.run()
    finally:
        # Terminate the mapping client.
        if mapping_client is not None:
            mapping_client.terminate()


if __name__ == "__main__":
    main()
