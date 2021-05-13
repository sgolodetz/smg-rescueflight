import cv2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import sys

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Tuple

from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLUtil, TriangleMesh
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotory.drones import SimulatedDrone


def render_scene(w_t_c: np.ndarray) -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def main() -> None:
    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Simulated Drone Client")

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Load the mesh.
    o3d_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(
        "C:/spaint/build/bin/apps/spaintgui/meshes/groundtruth.ply"
    )
    mesh: TriangleMesh = TriangleMesh(
        np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.vertex_colors), np.asarray(o3d_mesh.triangles)
    )

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.025
    )

    # Repeatedly:
    while True:
        # Process any PyGame events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # Allow the user to control the camera.
        camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

        # Clear the colour and depth buffers.
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Determine the viewing pose.
        viewing_pose: np.ndarray = camera_controller.get_pose()

        # Set the projection matrix.
        with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(intrinsics, *window_size)):
            # Set the model-view matrix.
            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                CameraPoseConverter.pose_to_modelview(viewing_pose)
            )):
                # Render a voxel grid.
                glColor3f(0.0, 0.0, 0.0)
                OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                # Render coordinate axes.
                CameraRenderer.render_camera(
                    CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
                )

                # TODO
                mesh.render()

        # Swap the front and back buffers.
        pygame.display.flip()

    # with SimulatedDrone(
    #     image_renderer=render_scene,
    #     image_size=(640, 480),
    #     intrinsics=(532.5694641250893, 531.5410880910171, 320.0, 240.0)
    # ) as drone:
    #     while True:
    #         image, w_t_c = drone.get_image_and_pose()
    #
    #         cv2.imshow("Image", image)
    #         cv2.waitKey(1)
    #
    #         print(w_t_c)
    #         print("===")


if __name__ == "__main__":
    main()
