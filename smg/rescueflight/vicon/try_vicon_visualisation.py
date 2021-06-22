import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

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
from smg.utility import FiducialUtil, GeometryUtil
from smg.vicon import SubjectFromSourceCache, ViconInterface, ViconSkeletonDetector


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
        "--scenes_folder", type=str, default="C:/spaint/build/bin/apps/spaintgui/meshes",
        help="the folder from which to load the scene mesh"
    )
    parser.add_argument(
        "--scene_timestamp", "-t", type=str,
        help="a timestamp indicating which scene mesh to load"
    )
    args: dict = vars(parser.parse_args())

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

    # Connect to the Vicon system.
    with ViconInterface() as vicon:
        # Construct the skeleton detector.
        skeleton_detector: ViconSkeletonDetector = ViconSkeletonDetector(vicon)

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
                        # Print out the frame number.
                        print(f"=== Frame {vicon.get_frame_number()} ===")

                        # TODO
                        skeletons: List[Skeleton3D] = skeleton_detector.detect_skeletons()
                        print(skeletons)

                        with SkeletonRenderer.default_lighting_context():
                            for skeleton_3d in skeletons:
                                SkeletonRenderer.render_skeleton(skeleton_3d)

                        # For each Vicon subject:
                        for subject in vicon.get_subject_names():
                            # Render all of its markers.
                            for marker_name, marker_pos in vicon.get_marker_positions(subject).items():
                                # print(marker_name, marker_pos)
                                glColor3f(1.0, 0.0, 0.0)
                                OpenGLUtil.render_sphere(marker_pos, 0.014, slices=10, stacks=10)

                            # Assume it's a single-segment subject and try to get its pose from the Vicon system.
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

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
