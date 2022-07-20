import numpy as np

from argparse import ArgumentParser
from typing import Optional

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingServer
from smg.dvmvs import DVMVSMonocularDepthEstimator
from smg.mapping.systems import OctomapMappingSystem
from smg.mvdepthnet import MVDepthMonocularDepthEstimator, MVDepth2MonocularDepthEstimator
from smg.utility import MonocularDepthEstimator, PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--batch", action="store_true",
        help="whether to run in batch mode"
    )
    parser.add_argument(
        "--camera_mode", "-m", type=str, choices=("follow", "free"), default="free",
        help="the camera mode"
    )
    parser.add_argument(
        "--depth_estimator_type", type=str, default="dvmvs", choices=("dvmvs", "mvdepth", "mvdepth2"),
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
        "--max_depth", type=float, default=3.0,
        help="the maximum depth values (in m) to keep during post-processing"
    )
    parser.add_argument(
        "--no_depth_postprocessing", action="store_true",
        help="whether to suppress depth post-processing"
    )
    parser.add_argument(
        "--octree_voxel_size", type=float, default=0.05,
        help="the voxel size (in m) to use for the Octomap"
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
        "--reconstruction_filename", type=str, default="octree.bt",
        help="the name of the file to which to save the reconstructed Octomap"
    )
    parser.add_argument(
        "--render_bodies", action="store_true",
        help="whether to render an SMPL body in place of each detected skeleton"
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="whether to save the sequence of frames used to reconstruct the Octomap"
    )
    parser.add_argument(
        "--save_people_masks", action="store_true",
        help="whether to save the people mask for each frame"
    )
    parser.add_argument(
        "--save_reconstruction", action="store_true",
        help="whether to save the reconstructed Octomap"
    )
    parser.add_argument(
        "--save_skeletons", action="store_true",
        help="whether to save the skeletons detected in each frame"
    )
    parser.add_argument(
        "--tsdf_voxel_size", type=float, default=0.01,
        help="the voxel size (in m) to use for the TSDF (if we're reconstructing it)"
    )
    parser.add_argument(
        "--use_arm_selection", action="store_true",
        help="whether to allow the user to select 3D points in the scene using their arm"
    )
    parser.add_argument(
        "--use_received_depth", action="store_true",
        help="whether to use depth images received from the client instead of estimating depth"
    )
    parser.add_argument(
        "--use_tsdf", action="store_true",
        help="whether to additionally reconstruct an Open3D TSDF for visualisation purposes"
    )
    args: dict = vars(parser.parse_args())

    depth_estimator_type: str = args.get("depth_estimator_type")
    output_dir: Optional[str] = args.get("output_dir")
    postprocess_depth: bool = not args.get("no_depth_postprocessing")

    # Construct the depth estimator.
    if depth_estimator_type == "dvmvs":
        depth_estimator: MonocularDepthEstimator = DVMVSMonocularDepthEstimator(max_depth=args["max_depth"])
    elif depth_estimator_type == "mvdepth":
        depth_estimator: MonocularDepthEstimator = MVDepthMonocularDepthEstimator(max_depth=args["max_depth"])
    else:
        depth_estimator: MonocularDepthEstimator = MVDepth2MonocularDepthEstimator(max_depth=args["max_depth"])

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.EPoolEmptyStrategy.make(args["pool_empty_strategy"])
    ) as server:
        # Construct the mapping system.
        with OctomapMappingSystem(
            server, depth_estimator, batch_mode=args["batch"], camera_mode=args["camera_mode"],
            detect_objects=args["detect_objects"], detect_skeletons=args["detect_skeletons"],
            max_received_depth=args["max_depth"], octree_voxel_size=args["octree_voxel_size"], output_dir=output_dir,
            postprocess_depth=postprocess_depth, reconstruction_filename=args["reconstruction_filename"],
            render_bodies=args["render_bodies"], save_frames=args["save_frames"],
            save_people_masks=args["save_people_masks"], save_reconstruction=args["save_reconstruction"],
            save_skeletons=args["save_skeletons"], tsdf_voxel_size=args["tsdf_voxel_size"],
            use_arm_selection=args["use_arm_selection"], use_received_depth=args["use_received_depth"],
            use_tsdf=args["use_tsdf"]
        ) as mapping_system:
            # Start the server.
            server.start()

            # Run the mapping system.
            mapping_system.run()


if __name__ == "__main__":
    main()
