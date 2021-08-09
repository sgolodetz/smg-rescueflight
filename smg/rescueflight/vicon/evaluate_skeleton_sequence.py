import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Skeleton3D, SkeletonRenderer, SkeletonUtil
from smg.utility import GeometryUtil, PoseUtil
from smg.vicon import OfflineViconInterface, ViconSkeletonDetector, ViconUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--detector_type", "-t", type=str, default="lcrnet", choices=("lcrnet", "xnect"),
        help="the skeleton detector whose (pre-saved) skeletons are to be evaluated"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the name of the directory containing the ground-truth Vicon sequence"
    )
    args: dict = vars(parser.parse_args())

    detector_type: str = args["detector_type"]
    sequence_dir: str = args["sequence_dir"]

    # TODO
    aruco_filename: str = os.path.join(sequence_dir, "reconstruction", "aruco_from_world.txt")
    if os.path.exists(aruco_filename):
        aruco_from_world: np.ndarray = PoseUtil.load_pose(aruco_filename)
    else:
        aruco_from_world: np.ndarray = np.eye(4)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Skeleton Sequence Evaluator")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05,
        canonical_linear_speed=0.1
    )

    # Connect to the Vicon interface.
    with OfflineViconInterface(folder=sequence_dir) as vicon:
        # Construct the ground-truth skeleton detector.
        gt_skeleton_detector: ViconSkeletonDetector = ViconSkeletonDetector(
            vicon, is_person=ViconUtil.is_person, use_vicon_poses=True
        )

        # Initialise the playback variables.
        pause: bool = True
        process_next: bool = True

        while True:
            # Process any PyGame events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # If the user presses the 'b' key, process the sequence without pausing.
                    if event.key == pygame.K_b:
                        pause = False
                        process_next = True

                    # Otherwise, if the user presses the 'n' key, process the next image and then pause.
                    elif event.key == pygame.K_n:
                        pause = True
                        process_next = True
                elif event.type == pygame.QUIT:
                    # If the user wants us to quit, shut down pygame.
                    pygame.quit()

                    # Then forcibly terminate the whole process.
                    # noinspection PyProtectedMember
                    os._exit(0)

            if process_next:
                vicon.get_frame()
                process_next = not pause

            # TODO
            frame_number: int = vicon.get_frame_number()

            print("===")
            print(f"Frame {frame_number}")

            gt_skeletons: Dict[str, Skeleton3D] = gt_skeleton_detector.detect_skeletons()
            detected_skeletons: Optional[List[Skeleton3D]] = SkeletonUtil.try_load_skeletons(
                os.path.join(sequence_dir, detector_type, f"{frame_number}.skeletons.txt")
            )

            # TODO
            # Look up the Vicon coordinate system positions of the all of the Vicon markers that can currently be seen
            # by the Vicon system, hopefully including ones for the ArUco marker corners.
            marker_positions: Dict[str, np.ndarray] = vicon.get_marker_positions("Registrar")

            # TODO
            if all(key in marker_positions for key in ["0_0", "0_1", "0_2", "0_3"]):
                offset: float = 0.0705  # 7.05cm (half the width of the printed marker)

                p: np.ndarray = np.column_stack([
                    marker_positions["0_0"],
                    marker_positions["0_1"],
                    marker_positions["0_2"],
                    marker_positions["0_3"]
                ])

                q: np.ndarray = np.array([
                    [-offset, -offset, 0],
                    [offset, -offset, 0],
                    [offset, offset, 0],
                    [-offset, offset, 0]
                ]).transpose()

                # Estimate the rigid transformation between the two sets of points.
                aruco_from_vicon: np.ndarray = GeometryUtil.estimate_rigid_transform(p, q)
            else:
                aruco_from_vicon: np.ndarray = np.eye(4)

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set the projection matrix.
            with OpenGLMatrixContext(
                GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix((500.0, 500.0, 320.0, 240.0), 640, 480)
            ):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
                )):
                    # Render a voxel grid.
                    glColor3f(0.0, 0.0, 0.0)
                    OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                    with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(aruco_from_vicon)):
                        # Render the ArUco marker in the location estimated for it by the Vicon system.
                        if all(key in marker_positions for key in ["0_0", "0_1", "0_2", "0_3"]):
                            glBegin(GL_QUADS)
                            glColor3f(0, 1, 0)
                            glVertex3f(*marker_positions["0_0"])
                            glVertex3f(*marker_positions["0_1"])
                            glVertex3f(*marker_positions["0_2"])
                            glVertex3f(*marker_positions["0_3"])
                            glEnd()

                    # Render the 3D skeletons.
                    with SkeletonRenderer.default_lighting_context():
                        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(aruco_from_vicon)):
                            for _, skeleton in gt_skeletons.items():
                                SkeletonRenderer.render_skeleton(skeleton)

                        if detected_skeletons is not None:
                            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(aruco_from_world)):
                                for skeleton in detected_skeletons:
                                    SkeletonRenderer.render_skeleton(skeleton)

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
