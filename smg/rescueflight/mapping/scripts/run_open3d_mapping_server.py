import cv2
import numpy as np
import open3d as o3d
import os

from argparse import ArgumentParser
from typing import List, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingServer
from smg.dvmvs import DVMVSMonocularDepthEstimator
from smg.mapping.systems import Open3DMappingSystem
from smg.mvdepthnet import MVDepthMonocularDepthEstimator
from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.relocalisation import ArUcoPnPRelocaliser
from smg.utility import MonocularDepthEstimator, PooledQueue, PoseUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--batch", action="store_true",
        help="whether to run in batch mode"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="whether to enable debugging"
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
        "--no_depth_postprocessing", action="store_true",
        help="whether to suppress depth post-processing"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        help="an optional directory into which to save the sequence"
    )
    parser.add_argument(
        "--pool_empty_strategy", "-p", type=str, default="replace_random",
        choices=("discard", "grow", "replace_random", "wait"),
        help="the strategy to use when a frame message is received whilst a client handler's frame pool is empty"
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="whether to save the sequence of frames used to reconstruct the TSDF"
    )
    parser.add_argument(
        "--save_reconstruction", action="store_true",
        help="whether to save the reconstructed mesh"
    )
    parser.add_argument(
        "--show_clean_mesh", action="store_true",
        help="whether to show a cleaned version of the mesh at the end of the process"
    )
    parser.add_argument(
        "--show_keyframes", action="store_true",
        help="whether to visualise the keyframes"
    )
    parser.add_argument(
        "--use_aruco_relocaliser", action="store_true",
        help="whether to use an ArUco+PnP relocaliser to align the map with a marker"
    )
    parser.add_argument(
        "--use_received_depth", action="store_true",
        help="whether to use depth images received from the client instead of estimating depth"
    )
    args: dict = vars(parser.parse_args())

    batch_mode: bool = args.get("batch")
    depth_estimator_type: str = args.get("depth_estimator_type")
    output_dir: Optional[str] = args.get("output_dir")
    postprocess_depth: bool = not args.get("no_depth_postprocessing")
    use_aruco_relocaliser: bool = args.get("use_aruco_relocaliser")

    # If requested, set up an ArUco+PnP relocaliser that can be used to align the map with a marker.
    aruco_relocaliser: Optional[ArUcoPnPRelocaliser] = None
    if use_aruco_relocaliser:
        offset: float = 0.0705  # 7.05cm (half the width of the printed marker)
        aruco_relocaliser = ArUcoPnPRelocaliser({
            "0_0": np.array([-offset, -offset, 0]),
            "0_1": np.array([offset, -offset, 0]),
            "0_2": np.array([offset, offset, 0]),
            "0_3": np.array([-offset, offset, 0])
        })

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
        mapping_system: Open3DMappingSystem = Open3DMappingSystem(
            server, depth_estimator, aruco_relocaliser=aruco_relocaliser, batch_mode=batch_mode, debug=args["debug"],
            detect_objects=args["detect_objects"], output_dir=output_dir, postprocess_depth=postprocess_depth,
            save_frames=args["save_frames"], use_received_depth=args["use_received_depth"]
        )

        # Start the server.
        server.start()

        # Run the mapping system.
        tsdf, objects = mapping_system.run()

        # Destroy any remaining OpenCV windows.
        cv2.destroyAllWindows()

        # Make a mesh from the TSDF.
        mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(tsdf, print_progress=True)

        # If we're not in batch mode, visualise the reconstructed map.
        if not batch_mode:
            grid: o3d.geometry.LineSet = VisualisationUtil.make_voxel_grid([-3, -2, -3], [3, 0, 3], [1, 1, 1])
            to_visualise: List[o3d.geometry.Geometry] = [grid, mesh]

            for obj in objects:
                box: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.AxisAlignedBoundingBox(*obj.box_3d)
                box.color = (1.0, 0.0, 1.0)
                to_visualise.append(box)

            if args["show_keyframes"]:
                keyframes: List[Tuple[np.ndarray, np.ndarray]] = depth_estimator.get_keyframes()
                to_visualise += [
                    VisualisationUtil.make_axes(pose, size=0.01) for _, pose in keyframes
                ]

            VisualisationUtil.visualise_geometries(to_visualise)

        # If an output directory has been specified and we're saving the reconstruction, save it now.
        if output_dir is not None and args["save_reconstruction"]:
            # Make sure the output directory exists.
            os.makedirs(output_dir, exist_ok=True)

            # Save the mesh itself.
            # noinspection PyTypeChecker
            o3d.io.write_triangle_mesh(os.path.join(output_dir, "mesh.ply"), mesh, print_progress=True)

            # Also save the transformation from world space to ArUco marker space if available.
            aruco_from_world: Optional[np.ndarray] = mapping_system.get_aruco_from_world()
            if aruco_from_world is not None:
                PoseUtil.save_pose(os.path.join(output_dir, "aruco_from_world.txt"), aruco_from_world)

        # If requested and we're not in batch mode, make and visualise a clean version of the mesh.
        if args["show_clean_mesh"] and not batch_mode:
            # See: http://www.open3d.org/docs/release/tutorial/geometry/mesh.html.
            triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
            triangle_clusters: np.ndarray = np.asarray(triangle_clusters)
            cluster_n_triangles: np.ndarray = np.asarray(cluster_n_triangles)
            triangles_to_remove: np.ndarray = cluster_n_triangles[triangle_clusters] < 500
            mesh.remove_triangles_by_mask(triangles_to_remove)
            VisualisationUtil.visualise_geometries([grid, mesh])


if __name__ == "__main__":
    main()
