import numpy as np
import os

from argparse import ArgumentParser
from typing import Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingServer
from smg.rescueflight.vicon.vicon_visualiser import ViconVisualiser
from smg.utility import PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--pause", action="store_true",
        help="whether to start the visualiser in its paused state"
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

        # Construct the visualiser.
        with ViconVisualiser(
            debug=False, mapping_server=mapping_server, pause=args["pause"],
            persistence_folder=persistence_folder, persistence_mode=persistence_mode,
            rendering_intrinsics=rendering_intrinsics, use_vicon_poses=args["use_vicon_poses"]
        ) as visualiser:
            # Run the visualiser.
            visualiser.run()
    finally:
        # Terminate the mapping server.
        if mapping_server is not None:
            mapping_server.terminate()


if __name__ == "__main__":
    main()
