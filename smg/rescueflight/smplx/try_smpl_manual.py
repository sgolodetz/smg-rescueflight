import math
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import smplx
import smplx.utils
import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from timeit import default_timer as timer

from smg.skeletons import SkeletonRenderer
from smg.smplx import SMPLBody


def draw_frame(body: SMPLBody) -> None:
    """
    Draw the current frame.

    :param body:    The SMPL body model.
    """
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(-3, 0, 3, 0, 0, 0, 0, 1, 0)

    glColor3f(0.75, 0.75, 0.75)

    with SkeletonRenderer.default_lighting_context():
        body.render()

    pygame.display.flip()


def main():
    # Initialise PyGame and create the window.
    pygame.init()
    window_size = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("SMPL Manual Pose Demo")

    # Load the SMPL body model.
    body: SMPLBody = SMPLBody("male")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Set the projection matrix.
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (window_size[0] / window_size[1]), 0.1, 1000.0)

    # Initialise the variables that control the animation.
    angle: float = 0.0
    direction: int = 0

    # Repeatedly:
    while True:
        # Process any PyGame events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # Update the animation variables.
        if direction == 0:
            if angle < 60.0:
                angle += 0.5
            else:
                angle = 60.0
                direction = 1
        else:
            if angle > 0.0:
                angle -= 0.5
            else:
                angle = 0.0
                direction = 0

        start = timer()

        # Manually set the local joint rotations based on the animation variables.
        body_pose: np.ndarray = np.zeros(smplx.SMPL.NUM_BODY_JOINTS * 3, dtype=np.float32)
        body_pose[3:6] = -np.array([1, 0, 0]) * angle * math.pi / 180
        body_pose[6:9] = np.array([1, 0, 0]) * angle * math.pi / 180
        body_pose[12:15] = np.array([1, 0, 0]) * angle * math.pi / 180
        body_pose[21:24] = np.array([1, 0, 0]) * angle / 2 * math.pi / 180
        body_pose[48:51] = -np.array([1, 0, 0]) * math.pi / 2
        body_pose[54:57] = np.array([0, 0, 1]) * math.pi / 2
        body.set_manual_pose(body_pose, np.eye(4))

        # Draw the current frame.
        draw_frame(body)

        end = timer()
        print("{}s".format(end - start))


if __name__ == "__main__":
    main()
