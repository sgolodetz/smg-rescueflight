import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, Optional, Tuple

from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.utility import PoseUtil
from smg.vicon import ViconInterface


class SubjectFromSourceCache:
    """A cache of the transformations from the spaces of the image sources to those of their Vicon subjects."""

    # CONSTRUCTOR

    def __init__(self, directory: str):
        self.__directory: str = directory
        self.__subjects_from_sources: Dict[str, np.ndarray] = {}

    # PUBLIC METHODS

    def get(self, subject_name: str) -> Optional[np.ndarray]:
        subject_from_source: Optional[np.ndarray] = self.__subjects_from_sources.get(subject_name)
        if subject_from_source is not None:
            return subject_from_source
        else:
            filename: str = os.path.join(self.__directory, f"subject_from_source-{subject_name}.txt")
            if os.path.exists(filename):
                subject_from_source = PoseUtil.load_pose(filename)
                self.__subjects_from_sources[subject_name] = subject_from_source
            else:
                return None


def main() -> None:
    np.set_printoptions(suppress=True)

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

                    # If a frame of Vicon data is available:
                    if vicon.get_frame():
                        # Print out the frame number.
                        print(f"=== Frame {vicon.get_frame_number()} ===")

                        # For each Vicon subject:
                        for subject in vicon.get_subject_names():
                            # Render all of its markers.
                            for marker_name, marker_pos in vicon.get_marker_positions(subject).items():
                                print(marker_name, marker_pos)
                                glColor3f(1.0, 0.0, 0.0)
                                OpenGLUtil.render_sphere(marker_pos, 0.014, slices=10, stacks=10)

                            # Assume it's a single-segment subject and try to render its coordinate axes.
                            subject_from_world: Optional[np.ndarray] = vicon.get_segment_pose(subject, subject)
                            if subject_from_world is not None:
                                subject_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(subject_from_world)
                                # glLineWidth(5)
                                CameraRenderer.render_camera(subject_cam, axis_scale=0.5)
                                # glLineWidth(1)

                                subject_from_source: Optional[np.ndarray] = subject_from_source_cache.get(subject)
                                if subject_from_source is not None:
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
