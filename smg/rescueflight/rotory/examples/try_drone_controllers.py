import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import cast, Dict, List, Optional, Tuple

from smg.meshing import MeshUtil
from smg.navigation import Path
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.pyoctomap import CM_COLOR_HEIGHT, OctomapUtil, OcTree, OcTreeDrawer
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotorcontrol import DroneControllerFactory
from smg.rotorcontrol.controllers import DroneController, TraverseWaypointsDroneController
from smg.rotory.drones import SimulatedDrone
from smg.skeletons import SkeletonRenderer


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_controller_type", type=str, default="keyboard",
        choices=("futaba_t6k", "keyboard", "traverse_waypoints"),
        help="the type of drone controller to use"
    )
    parser.add_argument(
        "--planning_octree", type=str,
        help="the name of the planning octree file (if any)"
    )
    parser.add_argument(
        "--scene_octree", type=str,
        help="the name of the scene octree file (if any)"
    )
    args: dict = vars(parser.parse_args())

    drone_controller_type: str = args.get("drone_controller_type")
    planning_octree_filename: str = args.get("planning_octree")
    scene_octree_filename: str = args.get("scene_octree")

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

    # Load in the drone mesh.
    drone_mesh: OpenGLTriMesh = MeshUtil.convert_trimesh_to_opengl(MeshUtil.load_tello_mesh())

    # Construct the drone.
    with SimulatedDrone(angular_gain=0.04) as drone:
        # Set the drone origin.
        drone_origin: SimpleCamera = SimpleCamera(
            np.array([1.5, 8.5, 12.5]) * 0.1, [0, 0, 1], [0, -1, 0]
            # np.array([-10.5, 0.5, 0.5]) * 0.1, [0, 0, 1], [0, -1, 0]
        )
        # drone.set_drone_origin(drone_origin)
        time.sleep(0.1)

        # Load the planning octree (if specified).
        planning_octree: Optional[OcTree] = None
        if planning_octree_filename is not None:
            planning_voxel_size: float = 0.1
            planning_octree = OcTree(planning_voxel_size)
            planning_octree.read_binary(planning_octree_filename)
        elif drone_controller_type == "traverse_waypoints":
            raise RuntimeError("A planning octree must be provided for 'traverse waypoints' control to be used")

        # Load the scene octree (if specified).
        # FIXME: This code duplicates the above - fix this before merging.
        scene_octree: Optional[OcTree] = None
        if scene_octree_filename is not None:
            scene_voxel_size: float = 0.1
            scene_octree = OcTree(scene_voxel_size)
            scene_octree.read_binary(scene_octree_filename)
        elif drone_controller_type == "traverse_waypoints":
            raise RuntimeError("A scene octree must be provided for 'traverse waypoints' control to be used")

        # Set up the octree drawer.
        drawer: OcTreeDrawer = OcTreeDrawer()
        drawer.set_color_mode(CM_COLOR_HEIGHT)

        # Construct the drone controller.
        kwargs: Dict[str, dict] = {
            "futaba_t6k": dict(drone=drone),
            "keyboard": dict(drone=drone),
            "traverse_waypoints": dict(drone=drone, planning_octree=planning_octree)
        }

        drone_controller: DroneController = DroneControllerFactory.make_drone_controller(
            drone_controller_type, **kwargs[drone_controller_type]
        )

        # If we're using 'traverse waypoints' control, tell the drone to take off and set some dummy waypoints.
        if drone_controller_type == "traverse_waypoints":
            drone.takeoff()

            cast(TraverseWaypointsDroneController, drone_controller).set_waypoints([
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

            drone_image, camera_w_t_c, chassis_w_t_c = drone.get_image_and_poses()

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Allow the user to control the drone.
            if True:  # drone_controller_type != "traverse_waypoints" or drone.get_state() == SimulatedDrone.FLYING:
                drone_controller.iterate(
                    events=events, image=drone_image, intrinsics=drone.get_intrinsics(),
                    tracker_c_t_i=np.linalg.inv(camera_w_t_c)
                )

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
                    if drone_controller_type == "traverse_waypoints":
                        traverse_waypoints_drone_controller = cast(TraverseWaypointsDroneController, drone_controller)
                        interpolated_path: Optional[Path] = traverse_waypoints_drone_controller.get_interpolated_path()
                        path: Optional[Path] = traverse_waypoints_drone_controller.get_path()
                        if path is not None:
                            path.render(
                                start_colour=(0, 1, 1), end_colour=(0, 1, 1), width=5,
                                waypoint_colourer=traverse_waypoints_drone_controller.get_occupancy_colourer()
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

            # Swap the front and back buffers.
            pygame.display.flip()

            # TODO
            time.sleep(0.01)


if __name__ == "__main__":
    main()
