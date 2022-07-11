import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from typing import Tuple

from smg.opengl import OpenGLFrameBuffer, OpenGLUtil


class Renderer:
    """A renderer."""

    def __init__(self):
        """Construct a renderer."""
        self.__framebuffer: OpenGLFrameBuffer = OpenGLFrameBuffer(640, 480)

    def make_image(self) -> np.ndarray:
        """
        Make a trivial image using the renderer's frame-buffer.

        :return:    The image.
        """
        with self.__framebuffer:
            # Clear the colour and depth buffers. The colour buffer is cleared to red so that the change can be seen.
            glClearColor(1.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Read and return the contents of the colour buffer (i.e. a red image).
            return OpenGLUtil.read_bgr_image(640, 480)


def main():
    # Initialise PyGame and create a hidden window so that we can use OpenGL.
    pygame.init()
    window_size: Tuple[int, int] = (1, 1)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.HIDDEN | pygame.OPENGL)

    # Construct the renderer.
    renderer: Renderer = Renderer()

    # Repeatedly render a simple image to the renderer's frame-buffer, and show it. Exit if 'q' is pressed.
    while True:
        cv2.imshow("Image", renderer.make_image())
        c: int = cv2.waitKey(1)
        if c == ord('q'):
            break


if __name__ == "__main__":
    main()
