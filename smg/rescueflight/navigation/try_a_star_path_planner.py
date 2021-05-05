import math
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import sys

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Optional, Tuple

from smg.navigation import AStarPathPlanner, PathUtil
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLUtil
from smg.pyoctomap import *
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil


def main() -> None:
    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("A* Path Planner")

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Set up the octree drawer.
    drawer: OcTreeDrawer = OcTreeDrawer()
    # drawer.set_color_mode(CM_COLOR_HEIGHT)
    # drawer.enable_freespace()

    # Create the octree.
    voxel_size: float = 0.05
    half_voxel_size: float = voxel_size / 2.0

    tree: OcTree = OcTree(voxel_size)

    origin: Vector3 = Vector3(half_voxel_size, half_voxel_size, half_voxel_size)
    offset: Vector3 = Vector3(voxel_size * 10, 0.0, 0.0)

    for angle in np.linspace(0.0, 2 * math.pi, 128, endpoint=False):
        angled_offset: Vector3 = offset.copy()
        angled_offset.rotate_ip(0, -angle, 0)
        tree.insert_ray(origin, origin + angled_offset)

    planner: AStarPathPlanner = AStarPathPlanner(tree, PathUtil.neighbours8)
    source = np.array([0.5, 0.5, 0.5]) * voxel_size
    goal = np.array([7.5, 0.5, 3.5]) * voxel_size

    start = timer()
    path: Optional[np.ndarray] = planner.plan_path(source=source, goal=goal)
    end = timer()
    print(f"Path Planning: {end - start}s")

    smoothed_path: Optional[np.ndarray] = PathUtil.interpolate(PathUtil.pull_strings(path, tree)) \
        if path is not None else None

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
                # Render coordinate axes.
                glLineWidth(5)
                CameraRenderer.render_camera(
                    CameraUtil.make_default_camera(), axis_scale=0.1, body_colour=(1.0, 1.0, 0.0), body_scale=0.01
                )
                glLineWidth(1)

                # Render a voxel grid.
                ten_voxel_size: float = voxel_size  # * 10
                glColor3f(0.0, 0.0, 0.0)
                OpenGLUtil.render_voxel_grid(
                    # [-2, -ten_voxel_size, -2], [2, ten_voxel_size, 2],
                    [-2, 0, -2], [2, ten_voxel_size, 2],
                    [ten_voxel_size, ten_voxel_size, ten_voxel_size],
                    dotted=False
                )

                # Draw the octree.
                OctomapUtil.draw_octree(tree, drawer)

                # If a path was found, draw it.
                if path is not None:
                    OpenGLUtil.render_path(smoothed_path, colour=(1, 0, 1), width=5)

        # Swap the front and back buffers.
        pygame.display.flip()


if __name__ == "__main__":
    main()
