import numpy as np
import open3d as o3d
import os

from argparse import ArgumentParser
from typing import Dict, List, Optional

from smg.open3d import VisualisationUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import FiducialUtil, MarkerUtil, PoseUtil


# noinspection PyUnusedLocal
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
        "--reconstruction_filename", "-r", type=str, default="smglib-20210805-105559.ply",
        help="the name of the file containing the reconstructed mesh to be evaluated"
    )
    parser.add_argument(
        "--reconstruction_render_style", type=str, choices=("hidden", "normal", "uniform"), default="uniform",
        help="the rendering style to use for the reconstructed mesh"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str,
        help="a directory containing a sequence whose reconstruction we want to evaluate"
    )
    args: dict = vars(parser.parse_args())

    sequence_dir: Optional[str] = args.get("sequence_dir")

    # Initialise some variables.
    gt_filename: Optional[str] = None
    initial_cam: Optional[SimpleCamera] = None
    output_gt_filename: Optional[str] = None
    output_reconstruction_filename: Optional[str] = None
    reconstruction_filename: Optional[str] = None
    target_from_gt: Optional[np.ndarray] = None
    target_from_world_filename: Optional[str] = None

    # If a sequence directory has been specified:
    if sequence_dir is not None:
        # Specify the relevant filenames based on the sequence directory, overriding those on the command line.
        gt_filename = os.path.join(sequence_dir, "gt", "world_mesh.ply")
        output_gt_filename = os.path.join(sequence_dir, "gt", "vicon_mesh.ply")
        output_reconstruction_filename = os.path.join(sequence_dir, "reconstruction", "vicon_mesh.ply")
        reconstruction_filename = os.path.join(sequence_dir, "reconstruction", "world_mesh.ply")
        target_from_world_filename = os.path.join(sequence_dir, "reconstruction", "vicon_from_world.txt")

        # Determine the transformation needed to transform the ground-truth mesh into Vicon space.
        target_from_gt = PoseUtil.load_pose(target_from_world_filename)

        # Set the initial camera to point along the y axis (since that's the horizontal axis in Vicon space).
        initial_cam = SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1])

    # Otherwise:
    else:
        # Treat the filenames specified on the command line as being relative to SemanticPaint's meshes folder.
        folder: str = "C:/spaint/build/bin/apps/spaintgui/meshes"

        gt_filename = os.path.join(folder, args["gt_filename"])
        output_gt_filename = os.path.join(folder, args["output_gt_filename"]) \
            if args["output_gt_filename"] is not None else None
        output_reconstruction_filename = os.path.join(folder, args["output_reconstruction_filename"]) \
            if args["output_reconstruction_filename"] is not None else None
        reconstruction_filename = os.path.join(folder, args["reconstruction_filename"])
        target_from_world_filename = os.path.join(folder, args["aruco_from_world_filename"])

        # Determine the transformation needed to transform the ground-truth mesh into ArUco space.
        gt_marker_positions: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(
            os.path.join(folder, args["fiducials_filename"])
        )
        target_from_gt = MarkerUtil.estimate_space_to_marker_transform(gt_marker_positions)

        # Set the initial camera to point along the z axis (since that's the horizontal axis in ArUco space).
        initial_cam = SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0])

    # If we determined how to transform the ground-truth mesh, read it in and transform it into the target space.
    if target_from_gt is not None:
        # noinspection PyUnresolvedReferences
        gt_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(gt_filename)
        gt_mesh = gt_mesh.transform(target_from_gt)

    # Otherwise, raise an exception. This can happen if some of the necessary marker positions weren't available.
    else:
        raise RuntimeError("Couldn't estimate transformation from ground-truth space to target space")

    # Read in the (metric) mesh we want to evaluate, and likewise transform it into the target space.
    # noinspection PyUnresolvedReferences
    reconstruction_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(reconstruction_filename)
    target_from_world: np.ndarray = PoseUtil.load_pose(target_from_world_filename)
    reconstruction_mesh.transform(target_from_world)

    # If requested, save the transformed meshes to disk for later use.
    if output_gt_filename is not None:
        # noinspection PyTypeChecker, PyUnresolvedReferences
        o3d.io.write_triangle_mesh(output_gt_filename, gt_mesh)

    if output_reconstruction_filename is not None:
        # noinspection PyTypeChecker, PyUnresolvedReferences
        o3d.io.write_triangle_mesh(output_reconstruction_filename, reconstruction_mesh)

    # Visualise the meshes to allow them to be compared.
    # noinspection PyUnresolvedReferences
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
