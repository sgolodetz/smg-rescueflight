import numpy as np

from argparse import ArgumentParser
from typing import Optional

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingServer
from smg.dvmvs import DVMVSMonocularDepthEstimator
from smg.mapping.mvdepth import MVDepthOctomapMappingSystem
from smg.mvdepthnet import MVDepthMonocularDepthEstimator
from smg.utility import MonocularDepthEstimator, PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--camera_mode", "-m", type=str, choices=("follow", "free"), default="free",
        help="the camera mode"
    )
    parser.add_argument(
        "--depth_estimator_type", type=str, default="dvmvs", choices=("dvmvs", "mvdepth"),
        help="the type of depth estimator to use"
    )
    parser.add_argument(
        "--detect_objects", "-d", action="store_true",
        help="whether to detect 3D objects"
    )
    parser.add_argument(
        "--detect_skeletons", action="store_true",
        help="whether to detect 3D skeletons"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        help="an optional directory into which to save output files"
    )
    parser.add_argument(
        "--pool_empty_strategy", "-p", type=str, default="replace_random",
        choices=("discard", "grow", "replace_random", "wait"),
        help="the strategy to use when a frame message is received whilst a client handler's frame pool is empty"
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="whether to save the sequence of frames used to reconstruct the Octomap"
    )
    parser.add_argument(
        "--save_skeletons", action="store_true",
        help="whether to save the skeletons detected in each frame"
    )
    parser.add_argument(
        "--save_reconstruction", action="store_true",
        help="whether to save the reconstructed Octomap"
    )
    parser.add_argument(
        "--use_arm_selection", action="store_true",
        help="whether to allow the user to select 3D points in the scene using their arm"
    )
    parser.add_argument(
        "--use_received_depth", action="store_true",
        help="whether to use depth images received from the client instead of estimating depth"
    )
    args: dict = vars(parser.parse_args())

    depth_estimator_type: str = args.get("depth_estimator_type")
    output_dir: Optional[str] = args["output_dir"]

    # Construct the depth estimator.
    if depth_estimator_type == "dvmvs":
        depth_estimator: MonocularDepthEstimator = DVMVSMonocularDepthEstimator()
    else:
        depth_estimator: MonocularDepthEstimator = MVDepthMonocularDepthEstimator(
            "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar", debug=args["debug"]
        )

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.EPoolEmptyStrategy.make(args["pool_empty_strategy"])
    ) as server:
        # Construct the mapping system.
        with MVDepthOctomapMappingSystem(
            server, depth_estimator, camera_mode=args["camera_mode"], detect_objects=args["detect_objects"],
            detect_skeletons=args["detect_skeletons"], output_dir=output_dir, save_frames=args["save_frames"],
            save_reconstruction=args["save_reconstruction"], save_skeletons=args["save_skeletons"],
            use_arm_selection=args["use_arm_selection"], use_received_depth=args["use_received_depth"]
        ) as mapping_system:
            # Start the server.
            server.start()

            # Run the mapping system.
            mapping_system.run()


if __name__ == "__main__":
    main()
