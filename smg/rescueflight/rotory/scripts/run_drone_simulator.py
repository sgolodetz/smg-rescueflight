import math
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import sys

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.joysticks import FutabaT6K
from smg.opengl import CameraRenderer, OpenGLFrameBuffer, OpenGLMatrixContext
from smg.opengl import OpenGLImageRenderer, OpenGLUtil, TriangleMesh
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotory.drones import SimulatedDrone
from smg.utility import ImageUtil


class MeshRenderer:
    """A simple OpenGL renderer that can render a triangle mesh either to the screen or to an image."""

    # CONSTRUCTOR

    def __init__(self, mesh: TriangleMesh, *, light_dirs: Optional[List[np.ndarray]] = None):
        """
        Construct an OpenGL renderer for a triangle mesh.

        :param mesh:        The triangle mesh.
        :param light_dirs:  The directions from which to light the mesh with directional lights.
        """
        self.__framebuffer = None  # type: Optional[OpenGLFrameBuffer]
        self.__mesh = mesh         # type: TriangleMesh

        if light_dirs is None:
            pos = np.array([0.0, -2.0, -1.0, 0.0])  # type: np.ndarray
            self.__light_dirs = [pos, -pos]         # type: List[np.ndarray]
        elif len(light_dirs) <= 8:
            self.__light_dirs = light_dirs          # type: List[np.ndarray]
        else:
            raise RuntimeError("At most 8 light directions can be specified")

    # PUBLIC METHODS

    def render(self) -> None:
        """Render the mesh."""
        # Save various attributes so that they can be restored later.
        glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT)

        # Enable lighting.
        glEnable(GL_LIGHTING)

        # Set up the directional lights.
        for i in range(len(self.__light_dirs)):
            light_idx = GL_LIGHT0 + i  # type: int
            glEnable(light_idx)
            glLightfv(light_idx, GL_DIFFUSE, np.array([1, 1, 1, 1]))
            glLightfv(light_idx, GL_SPECULAR, np.array([1, 1, 1, 1]))
            glLightfv(light_idx, GL_POSITION, self.__light_dirs[i])

        # Enable colour-based materials (i.e. let material properties be defined by glColor).
        glEnable(GL_COLOR_MATERIAL)

        # Enable back-face culling.
        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)

        # Enable depth testing.
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST)

        # Render the mesh itself.
        self.__mesh.render()

        # Restore the attributes to their previous states.
        glPopAttrib()

    def render_to_image(self, world_from_camera: np.ndarray, image_size: Tuple[int, int],
                        intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        # If the OpenGL framebuffer hasn't been constructed yet, construct it now.
        # FIXME: Support image size changes.
        width, height = image_size
        if self.__framebuffer is None:
            self.__framebuffer = OpenGLFrameBuffer(width, height)

        # TODO
        with self.__framebuffer:
            # Set the viewport to encompass the whole framebuffer.
            OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), (width, height))

            # Clear the background to black.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set the projection matrix.
            with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                intrinsics, width, height
            )):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(np.linalg.inv(world_from_camera))
                )):
                    self.render()
                    return OpenGLUtil.read_bgr_image(width, height)


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


def make_mesh_renderer(o3d_mesh: o3d.geometry.TriangleMesh) -> MeshRenderer:
    """
    Make an OpenGL mesh renderer from an Open3D triangle mesh.

    :param o3d_mesh:    The Open3D triangle mesh.
    :return:            The OpenGL mesh renderer.
    """
    o3d_mesh.compute_vertex_normals(True)
    mesh: TriangleMesh = TriangleMesh(
        np.asarray(o3d_mesh.vertices),
        np.asarray(o3d_mesh.vertex_colors),
        np.asarray(o3d_mesh.triangles),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals)
    )
    return MeshRenderer(mesh)


def render_window(*, drone_image: np.ndarray, drone_chassis_w_t_c: np.ndarray, image_renderer: OpenGLImageRenderer,
                  intrinsics: Tuple[float, float, float, float], scene_mesh_renderer: MeshRenderer,
                  tello_mesh_renderer: MeshRenderer, viewing_pose: np.ndarray, window_size: Tuple[int, int]) -> None:
    """
    TODO

    :param drone_image:         TODO
    :param drone_chassis_w_t_c: TODO
    :param image_renderer:      TODO
    :param intrinsics:          TODO
    :param scene_mesh_renderer: TODO
    :param tello_mesh_renderer: TODO
    :param viewing_pose:        TODO
    :param window_size:         TODO
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
            scene_mesh_renderer.render()

            # Render the mesh for the drone (at its current pose).
            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(drone_chassis_w_t_c)):
                tello_mesh_renderer.render()

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
    scene_mesh_renderer: MeshRenderer = make_mesh_renderer(
        o3d.io.read_triangle_mesh("C:/spaint/build/bin/apps/spaintgui/meshes/groundtruth-decimated.ply")
    )

    tello_mesh_renderer: MeshRenderer = make_mesh_renderer(load_tello_mesh("C:/smglib/meshes/tello.ply"))

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.025
    )

    # Construct the simulated drone.
    with SimulatedDrone(
        image_renderer=scene_mesh_renderer.render_to_image,
        image_size=(640, 480),
        intrinsics=intrinsics
    ) as drone:
        # Construct the image renderer.
        with OpenGLImageRenderer() as image_renderer:
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

                # Get the drone's image and poses, and print out the pose of its camera.
                drone_image, drone_camera_w_t_c, drone_chassis_w_t_c = drone.get_image_and_poses()
                print(drone_camera_w_t_c)

                # Allow the user to control the camera.
                camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                # Render the contents of the window.
                render_window(
                    drone_image=drone_image,
                    drone_chassis_w_t_c=drone_chassis_w_t_c,
                    image_renderer=image_renderer,
                    intrinsics=intrinsics,
                    scene_mesh_renderer=scene_mesh_renderer,
                    tello_mesh_renderer=tello_mesh_renderer,
                    viewing_pose=camera_controller.get_pose(),
                    window_size=window_size
                )


if __name__ == "__main__":
    main()
