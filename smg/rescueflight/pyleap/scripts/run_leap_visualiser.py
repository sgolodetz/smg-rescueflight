import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Tuple

from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.pyleap import leap, LeapController, LeapRenderer
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter


def main() -> None:
    np.set_printoptions(suppress=True)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Leap Visualiser")

    # Manually set the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the Leap controller.
    leap_controller: LeapController = LeapController()

    # Repeatedly:
    while True:
        # Process any PyGame events.
        for event in pygame.event.get():
            # If the user wants us to quit:
            if event.type == pygame.QUIT:
                # Shut down pygame, and destroy any OpenCV windows.
                pygame.quit()
                cv2.destroyAllWindows()

                # Forcibly terminate the whole process.
                # noinspection PyProtectedMember
                os._exit(0)

        # Get the current frame from the Leap.
        leap_frame: leap.Frame = leap_controller.frame()

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
                # Render a voxel grid.
                glColor3f(0.0, 0.0, 0.0)
                OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                # If the current frame from the Leap is valid, render any detected hands.
                if leap_frame.is_valid():
                    LeapRenderer.render_hands(leap_frame, leap_controller)

        # Swap the front and back buffers.
        pygame.display.flip()


if __name__ == "__main__":
    main()
