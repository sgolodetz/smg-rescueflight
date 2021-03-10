import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from typing import Tuple

from smg.opengl import OpenGLFrameBuffer


class Renderer:
    def __init__(self):
        self.__framebuffer: OpenGLFrameBuffer = OpenGLFrameBuffer(640, 480)

    def make_image(self) -> np.ndarray:
        with self.__framebuffer:
            # Clear the colour and depth buffers.
            glClearColor(1.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # TODO
            buffer = glReadPixels(0, 0, 640, 480, GL_BGR, GL_UNSIGNED_BYTE)
            return np.frombuffer(buffer, dtype=np.uint8).reshape((480, 640, 3))[::-1, :]


def main():
    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (1, 1)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.HIDDEN | pygame.OPENGL)

    renderer: Renderer = Renderer()
    while True:
        cv2.imshow("Image", renderer.make_image())
        c: int = cv2.waitKey(1)
        if c == ord('q'):
            break


if __name__ == "__main__":
    main()
