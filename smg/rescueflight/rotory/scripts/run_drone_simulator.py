import math
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import sys

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Tuple

from smg.joysticks import FutabaT6K
from smg.opengl import CameraRenderer, OpenGLImageRenderer, OpenGLMatrixContext, OpenGLPrespecifiedTriMeshRenderer
from smg.opengl import OpenGLTriMesh, OpenGLTriMeshRenderer, OpenGLUtil
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotory.drones import SimulatedDrone
from smg.utility import ImageUtil


def convert_trimesh_to_opengl(o3d_mesh: o3d.geometry.TriangleMesh) -> OpenGLTriMesh:
    """
    Convert an Open3D triangle mesh to an OpenGL one.

    :param o3d_mesh:    The Open3D triangle mesh.
    :return:            The OpenGL mesh.
    """
    o3d_mesh.compute_vertex_normals(True)
    return OpenGLTriMesh(
        np.asarray(o3d_mesh.vertices),
        np.asarray(o3d_mesh.vertex_colors),
        np.asarray(o3d_mesh.triangles),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals)
    )


# noinspection PyArgumentList
def load_tello_mesh(filename: str) -> o3d.geometry.TriangleMesh:
    """
    Load the DJI Tello mesh from the specified file.

    .. note::
        There is a special function for this because the mesh needs some processing to get it into a usable format.

    :param filename:    The name of the file containing the DJI Tello mesh.
    :return:            The DJI Tello mesh.
    """
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(filename)
    mesh.translate(-mesh.get_center())
    mesh.scale(0.002, np.zeros(3))
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([math.pi, 0, 0])))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array([0, 1, 1]))
    return mesh


def render_window(*, drone_chassis_w_t_c: np.ndarray, drone_image: np.ndarray, drone_mesh: OpenGLTriMesh,
                  image_renderer: OpenGLImageRenderer, intrinsics: Tuple[float, float, float, float],
                  mesh_renderer: OpenGLTriMeshRenderer, scene_mesh: OpenGLTriMesh, viewing_pose: np.ndarray,
                  window_size: Tuple[int, int]) -> None:
    """
    Render the application window.

    :param drone_chassis_w_t_c: The pose of the drone's chassis.
    :param drone_image:         The most recent image from the drone.
    :param drone_mesh:          The drone mesh.
    :param image_renderer:      An OpenGL-based image renderer.
    :param intrinsics:          The camera intrinsics.
    :param mesh_renderer:       An OpenGL-based mesh renderer.
    :param scene_mesh:          The scene mesh.
    :param viewing_pose:        The pose from which the scene is being viewed.
    :param window_size:         The application window size, as a (width, height) tuple.
    """
    # Clear the window.
    OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), window_size)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render the whole scene from the viewing pose.
    OpenGLUtil.set_viewport((0.0, 0.0), (0.5, 1.0), window_size)

    glPushAttrib(GL_DEPTH_BUFFER_BIT)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_DEPTH_TEST)

    with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(intrinsics, 640, 480)):
        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
            CameraPoseConverter.pose_to_modelview(viewing_pose)
        )):
            # Render a voxel grid.
            glColor3f(0.0, 0.0, 0.0)
            OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

            # Render coordinate axes.
            CameraRenderer.render_camera(CameraUtil.make_default_camera())

            # Render the mesh for the scene.
            mesh_renderer.render(scene_mesh)

            # Render the mesh for the drone (at its current pose).
            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(drone_chassis_w_t_c)):
                mesh_renderer.render(drone_mesh)

    glPopAttrib()

    # Render the drone image.
    OpenGLUtil.set_viewport((0.5, 0.0), (1.0, 1.0), window_size)
    image_renderer.render_image(ImageUtil.flip_channels(drone_image))

    # Swap the front and back buffers.
    pygame.display.flip()


def main() -> None:
    # Initialise pygame and some of its modules.
    pygame.init()
    pygame.joystick.init()
    pygame.mixer.init()

    # Make sure pygame always gets the user inputs.
    pygame.event.set_grab(True)

    # Try to determine the joystick index of the Futaba T6K. If no joystick is plugged in, early out.
    joystick_count: int = pygame.joystick.get_count()
    joystick_idx: int = 0
    if joystick_count == 0:
        exit(0)
    elif joystick_count != 1:
        # TODO: Prompt the user for the joystick to use.
        pass

    # Construct and calibrate the Futaba T6K.
    joystick: FutabaT6K = FutabaT6K(joystick_idx)
    joystick.calibrate()

    # Create the window.
    window_size: Tuple[int, int] = (1280, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Drone Simulator")

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Load in the meshes for the scene and the drone, and prepare them for rendering.
    scene_mesh: OpenGLTriMesh = convert_trimesh_to_opengl(
        o3d.io.read_triangle_mesh("C:/spaint/build/bin/apps/spaintgui/meshes/groundtruth-decimated.ply")
    )

    drone_mesh: OpenGLTriMesh = convert_trimesh_to_opengl(load_tello_mesh("C:/smglib/meshes/tello.ply"))

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.025
    )

    # Construct the image renderer.
    with OpenGLImageRenderer() as image_renderer:
        # Construct the triangle mesh renderer.
        with OpenGLTriMeshRenderer() as mesh_renderer:
            # Construct the simulated drone.
            with SimulatedDrone(
                image_renderer=OpenGLPrespecifiedTriMeshRenderer(scene_mesh, mesh_renderer).render_to_image,
                image_size=(640, 480), intrinsics=intrinsics
            ) as drone:
                # Load in the "drone flying" sound.
                pygame.mixer.music.load("C:/smglib/sounds/drone_flying.mp3")

                # Prevent the drone's gimbal from being moved until we're ready.
                can_move_gimbal: bool = False

                # Stop when both Button 0 and Button 1 on the Futaba T6K are set to their "released" state.
                while joystick.get_button(0) != 0 or joystick.get_button(1) != 0:
                    # Process any PyGame events.
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN:
                            # If Button 0 on the Futaba T6K is set to its "pressed" state:
                            if event.button == 0:
                                # Start playing the "drone flying" sound.
                                if drone.get_state() == SimulatedDrone.IDLE:
                                    pygame.mixer.music.play(loops=-1)

                                # Take off.
                                drone.takeoff()
                        elif event.type == pygame.JOYBUTTONUP:
                            # If Button 0 on the Futaba T6K is set to its "released" state, land.
                            if event.button == 0:
                                drone.land()
                        elif event.type == pygame.QUIT:
                            # If the user wants us to quit, do so.
                            pygame.quit()
                            sys.exit(0)

                    # If the drone is in the idle state, stop the "drone flying" sound.
                    if drone.get_state() == SimulatedDrone.IDLE:
                        pygame.mixer.music.stop()

                    # Update the movement of the drone based on the pitch, roll and yaw values output by the Futaba T6K.
                    drone.move_forward(joystick.get_pitch())
                    drone.turn(joystick.get_yaw())

                    if joystick.get_button(1) == 0:
                        drone.move_right(0)
                        drone.move_up(joystick.get_roll())
                    else:
                        drone.move_right(joystick.get_roll())
                        drone.move_up(0)

                    # If the throttle goes above half-way, enable movement of the drone's gimbal from now on.
                    throttle: float = joystick.get_throttle()
                    if throttle >= 0.5:
                        can_move_gimbal = True

                    # If the drone's gimbal can be moved, update its pitch based on the current value of the throttle.
                    # Note that the throttle value is in [0,1], so we rescale it to a value in [-1,1] as a first step.
                    if can_move_gimbal:
                        drone.update_gimbal_pitch(2 * (joystick.get_throttle() - 0.5))

                    # Get the drone's image and poses.
                    drone_image, drone_camera_w_t_c, drone_chassis_w_t_c = drone.get_image_and_poses()

                    # Allow the user to control the free-view camera.
                    camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                    # Render the contents of the window.
                    render_window(
                        drone_chassis_w_t_c=drone_chassis_w_t_c,
                        drone_image=drone_image,
                        drone_mesh=drone_mesh,
                        image_renderer=image_renderer,
                        intrinsics=intrinsics,
                        mesh_renderer=mesh_renderer,
                        scene_mesh=scene_mesh,
                        viewing_pose=camera_controller.get_pose(),
                        window_size=window_size
                    )


if __name__ == "__main__":
    main()
