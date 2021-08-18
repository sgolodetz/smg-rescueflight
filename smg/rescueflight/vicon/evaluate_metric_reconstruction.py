import numpy as np
import open3d as o3d
import os

from argparse import ArgumentParser
from typing import Dict, List, Optional

from smg.open3d import VisualisationUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import FiducialUtil, MarkerUtil, PoseUtil
from smg.vicon import OfflineViconInterface


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--aruco_from_world_filename", "-a", type=str, default="smglib-20210805-105559-aruco_from_world.txt",
        help="the name of the file containing the world space to ArUco space transformation"
    )
    parser.add_argument(
        "--fiducials_filename", "-f", type=str, default="TangoCapture-20210805-105559-fiducials.txt",
        help="the name of the fiducials file"
    )
    parser.add_argument(
        "--gt_filename", "-g", type=str, default="TangoCapture-20210805-105559-cleaned.ply",
        help="the name of the ground-truth mesh file"
    )
    parser.add_argument(
        "--gt_render_style", type=str, choices=("hidden", "normal", "uniform"), default="normal",
        help="the rendering style to use for the ground-truth mesh"
    )
    parser.add_argument(
        "--output_gt_filename", type=str,
        help="the name of the file to which to save the transformed ground-truth mesh (if any)"
    )
    parser.add_argument(
        "--output_reconstruction_filename", type=str,
        help="the name of the file to which to save the transformed reconstructed mesh (if any)"
    )
    parser.add_argument(
        "--paint_uniform", "-p", action="store_true",
        help="whether to paint the meshes uniform colours to make it easier to compare them"
    )
    parser.add_argument(
        "--reconstruction_filename", "-r", type=str, default="smglib-20210805-105559.ply",
        help="the name of the file containing the reconstructed mesh to be evaluated"
    )
    parser.add_argument(
        "--reconstruction_render_style", type=str, choices=("hidden", "normal", "uniform"), default="uniform",
        help="the rendering style to use for the reconstructed mesh"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str,
        help="a directory containing a sequence that we want to evaluate"
    )
    args: dict = vars(parser.parse_args())

    sequence_dir: Optional[str] = args.get("sequence_dir")
    target_from_gt: Optional[np.ndarray] = None

    # TODO: Comment here.
    if sequence_dir is not None:
        gt_filename: str = os.path.join(sequence_dir, "gt", "mesh.ply")
        output_gt_filename: Optional[str] = os.path.join(sequence_dir, "gt", "transformed_mesh.ply")
        output_reconstruction_filename: Optional[str] = os.path.join(
            sequence_dir, "reconstruction", "transformed_mesh.ply"
        )
        reconstruction_filename: str = os.path.join(sequence_dir, "reconstruction", "mesh.ply")
        target_from_world_filename: str = os.path.join(sequence_dir, "reconstruction", "vicon_from_world.txt")

        with OfflineViconInterface(folder=sequence_dir) as vicon:
            if vicon.get_frame():
                gt_marker_positions: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(
                    os.path.join(sequence_dir, "gt", "fiducials.txt")
                )
                target_from_gt = MarkerUtil.estimate_space_to_space_transform(
                    gt_marker_positions, vicon.get_marker_positions("Registrar")
                )

        initial_cam: SimpleCamera = SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1])

    # TODO: Comment here.
    else:
        folder: str = "C:/spaint/build/bin/apps/spaintgui/meshes"

        gt_filename: str = os.path.join(folder, args["gt_filename"])
        output_gt_filename: Optional[str] = os.path.join(folder, args["output_gt_filename"]) \
            if args["output_gt_filename"] is not None else None
        output_reconstruction_filename: Optional[str] = os.path.join(folder, args["output_reconstruction_filename"]) \
            if args["output_reconstruction_filename"] is not None else None
        reconstruction_filename: str = os.path.join(folder, args["reconstruction_filename"])
        target_from_world_filename: str = os.path.join(folder, args["aruco_from_world_filename"])

        gt_marker_positions: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(
            os.path.join(folder, args["fiducials_filename"])
        )
        target_from_gt = MarkerUtil.estimate_space_to_marker_transform(gt_marker_positions)

        initial_cam: SimpleCamera = SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0])

    # TODO: Comment here.
    if target_from_gt is None:
        raise RuntimeError("Couldn't estimate transformation from ground-truth space to target space")

    # Read in the (metric) mesh we want to evaluate, and transform it into the target space.
    reconstruction_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(reconstruction_filename)
    target_from_world: np.ndarray = PoseUtil.load_pose(target_from_world_filename)
    reconstruction_mesh.transform(target_from_world)

    # Read in the ground-truth mesh, and likewise transform it into the target space.
    gt_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(gt_filename)
    gt_mesh = gt_mesh.transform(target_from_gt)

    # If requested, save the transformed meshes to disk for later use.
    if output_gt_filename is not None:
        # noinspection PyTypeChecker
        o3d.io.write_triangle_mesh(output_gt_filename, gt_mesh)

    if output_reconstruction_filename is not None:
        # noinspection PyTypeChecker
        o3d.io.write_triangle_mesh(output_reconstruction_filename, reconstruction_mesh)

    # Visualise the meshes to allow them to be compared.
    geometries: List[o3d.geometry.Geometry] = [
        VisualisationUtil.make_axes(np.eye(4), size=0.25)
    ]

    if args["gt_render_style"] == "uniform":
        gt_mesh.paint_uniform_color((0.0, 1.0, 0.0))
    if args["gt_render_style"] != "hidden":
        geometries.append(gt_mesh)
    if args["reconstruction_render_style"] == "uniform":
        reconstruction_mesh.paint_uniform_color((1.0, 0.0, 0.0))
    if args["reconstruction_render_style"] != "hidden":
        geometries.append(reconstruction_mesh)

    VisualisationUtil.visualise_geometries(
        geometries, initial_pose=CameraPoseConverter.camera_to_pose(initial_cam), mesh_show_wireframe=True
    )


if __name__ == "__main__":
    main()
