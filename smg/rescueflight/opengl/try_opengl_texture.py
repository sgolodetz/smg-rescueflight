import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Tuple

from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLTexture, OpenGLTextureContext, OpenGLUtil
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("OpenGL Texture Test")

    # Manually set the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the texture and load in the texture image.
    texture: OpenGLTexture = OpenGLTexture()
    texture_image: np.ndarray = cv2.imread("D:/smplx/models/smpl/uv_texture.png")

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

        # Allow the user to control the camera.
        camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

        # Clear the colour and depth buffers.
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix.
        with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(intrinsics, *window_size)):
            # Set the model-view matrix.
            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
            )):
                # Render coordinate axes.
                CameraRenderer.render_camera(
                    CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
                )

                # Draw a textured triangle.
                with OpenGLTextureContext(texture):
                    texture.set_image(texture_image)

                    glColor3f(1.0, 1.0, 1.0)

                    glBegin(GL_TRIANGLES)
                    glTexCoord2f(0, 1)
                    glVertex3f(0, 0, 0)
                    glTexCoord2f(1, 1)
                    glVertex3f(1, 0, 0)
                    glTexCoord2f(0, 0)
                    glVertex3f(0, -1, 0)
                    glEnd()

        # Swap the front and back buffers.
        pygame.display.flip()


if __name__ == "__main__":
    main()
