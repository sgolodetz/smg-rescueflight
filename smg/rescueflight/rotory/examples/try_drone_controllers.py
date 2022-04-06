import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import cast, Dict, List, Optional, Tuple

from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLUtil
from smg.pyoctomap import OcTree
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotorcontrol import DroneControllerFactory
from smg.rotorcontrol.controllers import DroneController, FollowWaypointsDroneController
from smg.rotory.drones import SimulatedDrone
from smg.skeletons import SkeletonRenderer


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_controller_type", type=str, default="keyboard",
        choices=("follow_waypoints", "futaba_t6k", "keyboard"),
        help="the type of drone controller to use"
    )
    parser.add_argument(
        "--planning_octree", type=str,
        help="the name of the planning octree file (if any)"
    )
    args: dict = vars(parser.parse_args())

    drone_controller_type: str = args.get("drone_controller_type")
    planning_octree_filename: str = args.get("planning_octree")

    # Initialise PyGame and its joystick module.
    pygame.init()
    pygame.joystick.init()

    # Create the window.
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Drone Control Demo")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Construct the drone.
    with SimulatedDrone() as drone:
        # Load the planning octree (if specified).
        planning_octree: Optional[OcTree] = None
        if drone_controller_type == "follow_waypoints":
            if planning_octree_filename is not None:
                voxel_size: float = 0.1
                planning_octree = OcTree(voxel_size)
                planning_octree.read_binary(planning_octree_filename)
            else:
                raise RuntimeError("A planning octree must be provided for 'follow waypoints' control to be used")

        # Construct the drone controller.
        kwargs: Dict[str, dict] = {
            "follow_waypoints": dict(drone=drone, planning_octree=planning_octree),
            "futaba_t6k": dict(drone=drone),
            "keyboard": dict(drone=drone)
        }

        drone_controller: DroneController = DroneControllerFactory.make_drone_controller(
            drone_controller_type, **kwargs[drone_controller_type]
        )

        # If we're using 'follow waypoints' control, set some dummy waypoints.
        if drone_controller_type == "follow_waypoints":
            cast(FollowWaypointsDroneController, drone_controller).set_waypoints([
                np.array([30.5, 5.5, 5.5]) * 0.05
            ])

        # TODO
        while not drone_controller.should_quit():
            # Process any PyGame events.
            events: List[pygame.event.Event] = []
            for event in pygame.event.get():
                # Record the event for later use by the drone controller.
                events.append(event)

                # If the user wants us to quit:
                if event.type == pygame.QUIT:
                    # Shut down pygame.
                    pygame.quit()

                    # Forcibly terminate the whole process.
                    # noinspection PyProtectedMember
                    os._exit(0)

            drone_image, drone_w_t_c, _ = drone.get_image_and_poses()

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Allow the user to control the drone.
            drone_controller.iterate(events=events, image=drone_image, intrinsics=drone.get_intrinsics())

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

                    with SkeletonRenderer.default_lighting_context():
                        # TODO
                        glColor3f(0.0, 1.0, 0.0)
                        OpenGLUtil.render_sphere(drone_w_t_c[0:3, 3], 0.1, slices=10, stacks=10)

            # Swap the front and back buffers.
            pygame.display.flip()

            # TODO
            time.sleep(0.01)


if __name__ == "__main__":
    main()
