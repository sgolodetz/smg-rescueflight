import math
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import sys

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.navigation import AStarPathPlanner, OCS_OCCUPIED, PlanningToolkit
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
    drawer.set_color_mode(CM_COLOR_HEIGHT)
    # drawer.enable_freespace()

    # Load the octrees.
    planning_voxel_size: float = 0.1
    planning_tree: OcTree = OcTree(planning_voxel_size)
    planning_tree.read_binary("C:/smglib/smg-mapping/output-navigation/octree10cm.bt")

    rendering_voxel_size: float = 0.05
    rendering_tree: OcTree = OcTree(rendering_voxel_size)
    rendering_tree.read_binary("C:/smglib/smg-mapping/output-navigation/octree5cm.bt")

    # Construct the planning toolkit and the A* path planner.
    planning_toolkit: PlanningToolkit = PlanningToolkit(
        planning_tree,
        neighbours=PlanningToolkit.neighbours6,
        node_is_free=lambda n: planning_toolkit.occupancy_status(n) != OCS_OCCUPIED
    )

    planner: AStarPathPlanner = AStarPathPlanner(planning_toolkit, debug=True)

    # Specify the waypoints.
    waypoints: List[np.ndarray] = [
        np.array([0.5, 0.5, 5.5]) * rendering_voxel_size,
        np.array([-5.5, 10.5, 15.5]) * rendering_voxel_size,
        np.array([30.5, 5.5, 5.5]) * rendering_voxel_size,
        np.array([50.5, 0.5, 20.5]) * rendering_voxel_size
    ]

    # Plan a path through the waypoints.
    start = timer()
    ay: float = 10
    path: Optional[np.ndarray] = planner.plan_multipath(
        waypoints,
        d=PlanningToolkit.l1_distance(ay=ay),
        h=PlanningToolkit.l1_distance(ay=ay),
        allow_shortcuts=True,
        pull_strings=True,
        use_clearance=True
    )
    end = timer()
    print(f"Path Planning: {end - start}s")

    # Smooth any path found.
    start = timer()
    interpolated_path: Optional[np.ndarray] = PlanningToolkit.interpolate_path(path) if path is not None else None
    end = timer()
    print(f"Path Smoothing: {end - start}s")

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

                # Draw the octree.
                OctomapUtil.draw_octree(rendering_tree, drawer)

                # If a path was found, draw it.
                if path is not None:
                    OpenGLUtil.render_path(
                        interpolated_path, start_colour=(1, 1, 0), end_colour=(1, 0, 1), width=5,
                        waypoint_colourer=planning_toolkit.occupancy_colourer()
                    )

        # Swap the front and back buffers.
        pygame.display.flip()


if __name__ == "__main__":
    main()
