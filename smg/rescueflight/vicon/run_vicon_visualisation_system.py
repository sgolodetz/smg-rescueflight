import numpy as np
import os

from argparse import ArgumentParser
from typing import Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingServer
from smg.utility import PooledQueue

# FIXME: This shouldn't be in the current directory (it's not a package).
from vicon_visualisation_system import ViconVisualisationSystem


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--pause", action="store_true",
        help="whether to start the visualisation system in its paused state"
    )
    parser.add_argument(
        "--persistence_folder", type=str,
        help="the folder (if any) that should be used for Vicon persistence"
    )
    parser.add_argument(
        "--persistence_mode", type=str, default="none", choices=("input", "none", "output"),
        help="the Vicon persistence mode"
    )
    parser.add_argument(
        "--run_server", action="store_true",
        help="whether to accept connections from mapping clients"
    )
    parser.add_argument(
        "--scenes_folder", type=str, default="C:/spaint/build/bin/apps/spaintgui/meshes",
        help="the folder from which to load the scene mesh"
    )
    parser.add_argument(
        "--scene_timestamp", type=str,
        help="a timestamp indicating which scene mesh to load"
    )
    parser.add_argument(
        "--use_vicon_poses", action="store_true",
        help="whether to use the joint poses produced by the Vicon system"
    )
    args: dict = vars(parser.parse_args())

    persistence_folder: Optional[str] = args["persistence_folder"]
    persistence_mode: str = args["persistence_mode"]

    if persistence_mode != "none" and persistence_folder is None:
        raise RuntimeError(f"Cannot {persistence_mode}: need to specify a persistence folder")
    if persistence_mode == "input" and not os.path.exists(persistence_folder):
        raise RuntimeError("Cannot input: persistence folder does not exist")
    if persistence_mode == "output" and os.path.exists(persistence_folder):
        raise RuntimeError("Cannot output: persistence folder already exists")

    # Set the rendering intrinsics.
    rendering_intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Construct and start a mapping server if requested.
    mapping_server: Optional[MappingServer] = None
    try:
        if args["run_server"]:
            mapping_server = MappingServer(
                frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
                pool_empty_strategy=PooledQueue.PES_REPLACE_RANDOM
            )
            mapping_server.start()

        # Construct the visualisation system.
        with ViconVisualisationSystem(
            debug=False,
            mapping_server=mapping_server,
            pause=args["pause"] or args["run_server"],
            persistence_folder=persistence_folder,
            persistence_mode=persistence_mode,
            rendering_intrinsics=rendering_intrinsics,
            scene_timestamp=args["scene_timestamp"],
            scenes_folder=args["scenes_folder"],
            use_vicon_poses=args["use_vicon_poses"]
        ) as visualisation_system:
            # Run the visualisation system.
            visualisation_system.run()
    finally:
        # Terminate the mapping server.
        if mapping_server is not None:
            mapping_server.terminate()


if __name__ == "__main__":
    main()
