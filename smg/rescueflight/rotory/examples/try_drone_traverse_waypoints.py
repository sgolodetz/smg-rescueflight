import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.meshing import MeshUtil
from smg.navigation import Path
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.pyoctomap import CM_COLOR_HEIGHT, OctomapPicker, OctomapUtil, OcTree, OcTreeDrawer
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotorcontrol.controllers import TraverseWaypointsDroneController
from smg.rotory.drones import SimulatedDrone
from smg.skeletons import SkeletonRenderer


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--planning_octree", type=str, required=True,
        help="the name of the planning octree file"
    )
    parser.add_argument(
        "--scene_octree", type=str, required=True,
        help="the name of the scene octree file"
    )
    args: dict = vars(parser.parse_args())

    planning_octree_filename: str = args.get("planning_octree")
    scene_octree_filename: str = args.get("scene_octree")

    # Initialise PyGame and its joystick module.
    pygame.init()
    pygame.joystick.init()

    # Create the window.
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Traverse Waypoints Demo")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Load in the drone mesh.
    drone_mesh: OpenGLTriMesh = MeshUtil.convert_trimesh_to_opengl(MeshUtil.load_tello_mesh())

    # Construct the drone.
    with SimulatedDrone(angular_gain=0.08) as drone:
        # Load the planning octree.
        planning_voxel_size: float = 0.2
        planning_octree: OcTree = OcTree(planning_voxel_size)
        planning_octree.read_binary(planning_octree_filename)

        # Load the scene octree.
        scene_voxel_size: float = 0.2
        scene_octree: OcTree = OcTree(scene_voxel_size)
        scene_octree.read_binary(scene_octree_filename)

        # Set up the picker.
        # noinspection PyTypeChecker
        picker: OctomapPicker = OctomapPicker(scene_octree, *window_size, intrinsics)

        # Set up the octree drawer.
        drawer: OcTreeDrawer = OcTreeDrawer()
        drawer.set_color_mode(CM_COLOR_HEIGHT)

        # Construct the drone controller.
        drone_controller: TraverseWaypointsDroneController = TraverseWaypointsDroneController(
            drone=drone, planning_octree=planning_octree
        )

        # Tell the drone to take off.
        drone.takeoff()

        # TODO
        picker_pos: Optional[np.ndarray] = None

        # TODO
        while not drone_controller.should_quit():
            # Process any PyGame events.
            events: List[pygame.event.Event] = []
            for event in pygame.event.get():
                # Record the event for later use by the drone controller.
                events.append(event)

                # TODO
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    drone_controller.set_waypoints([picker_pos])

                # If the user wants us to quit:
                if event.type == pygame.QUIT:
                    # Shut down pygame.
                    pygame.quit()

                    # Forcibly terminate the whole process.
                    # noinspection PyProtectedMember
                    os._exit(0)

            drone_image, camera_w_t_c, chassis_w_t_c = drone.get_image_and_poses()

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Allow the user to control the drone.
            if drone.get_state() == SimulatedDrone.FLYING:
                drone_controller.iterate(
                    events=events, image=drone_image, intrinsics=drone.get_intrinsics(),
                    tracker_c_t_i=np.linalg.inv(camera_w_t_c)
                )

            # If there's a picker, pick from the viewing pose.
            picking_image, picking_mask = None, None
            if picker is not None:
                picking_image, picking_mask = picker.pick(np.linalg.inv(camera_controller.get_pose()))

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

                    # Render coordinate axes.
                    CameraRenderer.render_camera(
                        CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
                    )

                    # If we're using a scene octree, draw it.
                    if scene_octree is not None:
                        OctomapUtil.draw_octree(scene_octree, drawer)
                        # OctomapUtil.draw_octree_understandable(scene_octree, drawer, render_filled_cubes=False)

                    # TODO
                    interpolated_path: Optional[Path] = drone_controller.get_interpolated_path()
                    path: Optional[Path] = drone_controller.get_path()
                    if path is not None:
                        path.render(
                            start_colour=(0, 1, 1), end_colour=(0, 1, 1), width=5,
                            waypoint_colourer=drone_controller.get_occupancy_colourer()
                        )
                        # interpolated_path.render(
                        #     start_colour=(1, 1, 0), end_colour=(1, 0, 1), width=5,
                        #     waypoint_colourer=None
                        # )

                    # TODO
                    with SkeletonRenderer.default_lighting_context():
                        # Render the mesh for the drone (at its current pose).
                        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(camera_w_t_c)):
                            drone_mesh.render()

                    if picker is not None:
                        # Show the 3D cursor.
                        x, y = pygame.mouse.get_pos()
                        if picking_mask[y, x] != 0:
                            picker_pos = picking_image[y, x] + np.array([0, -0.5, 0])

                            glColor3f(0, 1, 0)
                            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                            OpenGLUtil.render_sphere(picker_pos, 0.1, slices=10, stacks=10)
                            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Swap the front and back buffers.
            pygame.display.flip()

            # TODO
            time.sleep(0.01)


if __name__ == "__main__":
    main()
