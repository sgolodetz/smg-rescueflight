import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

from smg.meshing import MeshUtil
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.skeletons import Skeleton3D, SkeletonRenderer
from smg.smplx import SMPLBody
from smg.utility import FiducialUtil, GeometryUtil
from smg.vicon import LiveViconInterface, OfflineViconInterface, SubjectFromSourceCache
from smg.vicon import ViconFrameSaver, ViconInterface, ViconSkeletonDetector


def is_person(subject_name: str) -> bool:
    """
    Determine whether or not the specified Vicon subject is a person.

    :param subject_name:    The name of the subject.
    :return:                True, if the specified Vicon subject is a person, or False otherwise.
    """
    return subject_name == "Madhu"


def load_scene_mesh(scenes_folder: str, scene_timestamp: str, vicon: ViconInterface) -> OpenGLTriMesh:
    """
    Load in a scene mesh, transforming it into the Vicon coordinate system in the process.

    :param scenes_folder:   The folder from which to load the scene mesh.
    :param scene_timestamp: A timestamp indicating which scene mesh to load.
    :param vicon:           The Vicon interface.
    :return:                The scene mesh.
    """
    # Specify the file paths.
    mesh_filename: str = os.path.join(scenes_folder, f"TangoCapture-{scene_timestamp}-cleaned.ply")
    fiducials_filename: str = os.path.join(scenes_folder, f"TangoCapture-{scene_timestamp}-fiducials.txt")

    # Load in the positions of the four ArUco marker corners as estimated during the reconstruction process.
    fiducials: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(fiducials_filename)

    # Stack these positions into a 3x4 matrix.
    p: np.ndarray = np.column_stack([
        fiducials["0_0"],
        fiducials["0_1"],
        fiducials["0_2"],
        fiducials["0_3"]
    ])

    # Look up the Vicon coordinate system positions of the all of the Vicon markers that can currently be seen
    # by the Vicon system, hopefully including ones for the ArUco marker corners.
    marker_positions: Dict[str, np.ndarray] = vicon.get_marker_positions("Registrar")

    # Again, stack the relevant positions into a 3x4 matrix.
    q: np.ndarray = np.column_stack([
        marker_positions["0_0"],
        marker_positions["0_1"],
        marker_positions["0_2"],
        marker_positions["0_3"]
    ])

    # Estimate the rigid transformation between the two sets of points.
    transform: np.ndarray = GeometryUtil.estimate_rigid_transform(p, q)

    # Load in the scene mesh and transform it into the Vicon coordinate system.
    scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_filename)
    scene_mesh_o3d.transform(transform)

    # Convert the scene mesh to OpenGL format and return it.
    return MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d)


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--persistence_folder", type=str,
        help="the folder (if any) that should be used for Vicon persistence"
    )
    parser.add_argument(
        "--persistence_mode", type=str, default="none", choices=("input", "none", "output"),
        help="the Vicon persistence mode"
    )
    parser.add_argument(
        "--scenes_folder", type=str, default="C:/spaint/build/bin/apps/spaintgui/meshes",
        help="the folder from which to load the scene mesh"
    )
    parser.add_argument(
        "--scene_timestamp", "-t", type=str,
        help="a timestamp indicating which scene mesh to load"
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

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Vicon Visualisation Demo")

    # Set the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the subject-from-source cache.
    subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")

    # Set the target frame time.
    target_frame_time: float = 1/30

    # Connect to the Vicon system.
    vicon: Optional[ViconInterface] = None

    try:
        if persistence_mode == "input":
            vicon = OfflineViconInterface(folder=persistence_folder)
        else:
            vicon = LiveViconInterface()

        # If we're in output mode, construct the frame saver.
        frame_saver: Optional[ViconFrameSaver] = None
        if persistence_mode == "output":
            frame_saver = ViconFrameSaver(folder=persistence_folder, vicon=vicon)

        # Construct the skeleton detector.
        skeleton_detector: ViconSkeletonDetector = ViconSkeletonDetector(vicon, is_person=is_person)

        # Load the SMPL body model.
        body: SMPLBody = SMPLBody("male")

        # Load in the scene mesh (if any), transforming it as needed in the process.
        scene_mesh: Optional[OpenGLTriMesh] = None
        scene_timestamp: Optional[str] = args.get("scene_timestamp")
        if scene_timestamp is not None and vicon.get_frame():
            scene_mesh = load_scene_mesh(args["scenes_folder"], scene_timestamp, vicon)

        # Repeatedly:
        while True:
            # Process any PyGame events.
            for event in pygame.event.get():
                # If the user wants us to quit:
                if event.type == pygame.QUIT:
                    # Shut down pygame.
                    pygame.quit()

                    # Forcibly terminate the whole process.
                    # noinspection PyProtectedMember
                    os._exit(0)

            # Work out the earliest desired end time for the frame.
            delay_until: float = timer() + target_frame_time

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set the projection matrix.
            with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                intrinsics, *window_size
            )):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
                )):
                    # Render coordinate axes.
                    CameraRenderer.render_camera(
                        CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
                    )

                    # Render a voxel grid.
                    glColor3f(0.0, 0.0, 0.0)
                    OpenGLUtil.render_voxel_grid([-3, -5, 0], [3, 5, 2], [1, 1, 1], dotted=True)

                    # Render the scene mesh (if any).
                    if scene_mesh is not None:
                        scene_mesh.render()

                    # If a frame of Vicon data is available:
                    if vicon.get_frame():
                        # If we're in output mode, save the frame to disk.
                        if persistence_mode == "output":
                            frame_saver.save_frame()

                        # Print out the frame number.
                        print(f"=== Frame {vicon.get_frame_number()} ===")

                        # For each Vicon subject:
                        for subject in vicon.get_subject_names():
                            # Render all of its markers.
                            for marker_name, marker_pos in vicon.get_marker_positions(subject).items():
                                glColor3f(1.0, 0.0, 0.0)
                                OpenGLUtil.render_sphere(marker_pos, 0.014, slices=10, stacks=10)

                            # If the subject is a person, don't bother trying to render its (rigid-body) pose.
                            if is_person(subject):
                                continue

                            # Otherwise, assume it's a single-segment subject and try to get its pose.
                            subject_from_world: Optional[np.ndarray] = vicon.get_segment_pose(subject, subject)

                            # If that succeeds:
                            if subject_from_world is not None:
                                # Render the subject pose obtained from the Vicon system.
                                subject_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(subject_from_world)
                                CameraRenderer.render_camera(subject_cam, axis_scale=0.5)

                                # Assume that the subject corresponds to an image source, and try to get the
                                # relative transformation from that image source to the subject.
                                subject_from_source: Optional[np.ndarray] = subject_from_source_cache.get(subject)

                                # If that succeeds (i.e. it does correspond to an image source, and we know the
                                # relative transformation):
                                if subject_from_source is not None:
                                    # Render the pose of the image source as well.
                                    source_from_world: np.ndarray = \
                                        np.linalg.inv(subject_from_source) @ subject_from_world
                                    source_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(source_from_world)
                                    glLineWidth(5)
                                    CameraRenderer.render_camera(source_cam, axis_scale=0.5)
                                    glLineWidth(1)

                        # Detect any skeletons in the frame.
                        skeletons: List[Skeleton3D] = skeleton_detector.detect_skeletons()

                        # Render the skeletons and their corresponding SMPL bodies.
                        with SkeletonRenderer.default_lighting_context():
                            for skeleton in skeletons:
                                SkeletonRenderer.render_skeleton(skeleton)
                                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                                body.render_from_skeleton(skeleton)
                                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                                SkeletonRenderer.render_keypoint_orienters(skeleton)

            # Swap the front and back buffers.
            pygame.display.flip()

            # Wait before moving onto the next frame if necessary.
            delay: float = delay_until - timer()
            if delay > 0:
                time.sleep(delay)
    finally:
        # Terminate the Vicon system.
        if vicon is not None:
            vicon.terminate()


if __name__ == "__main__":
    main()
