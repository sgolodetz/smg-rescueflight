import math
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import sys

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Optional, Tuple

from smg.joysticks import FutabaT6K
from smg.opengl import CameraRenderer, OpenGLFrameBuffer, OpenGLMatrixContext
from smg.opengl import OpenGLImageRenderer, OpenGLUtil, TriangleMesh
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotory.drones import SimulatedDrone
from smg.utility import ImageUtil


class MeshRenderer:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, mesh: TriangleMesh):
        self.__framebuffer: Optional[OpenGLFrameBuffer] = None
        self.__mesh: TriangleMesh = mesh

    # PUBLIC METHODS

    def render(self) -> None:
        # Enable lighting.
        glEnable(GL_LIGHTING)

        # Set up the first directional light.
        glEnable(GL_LIGHT0)
        pos = np.array([0.0, -2.0, -1.0, 0.0])  # type: np.ndarray
        glLightfv(GL_LIGHT0, GL_POSITION, pos)

        # Set up the second directional light.
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, np.array([1, 1, 1, 1]))
        glLightfv(GL_LIGHT1, GL_SPECULAR, np.array([1, 1, 1, 1]))
        glLightfv(GL_LIGHT1, GL_POSITION, -pos)

        # Enable colour-based materials (i.e. let material properties be defined by glColor).
        glEnable(GL_COLOR_MATERIAL)

        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)

        # TODO
        self.__mesh.render()

        glDisable(GL_CULL_FACE)

        # Disable colour-based materials and lighting again.
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHTING)

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
    Load a DJI Tello mesh from the specified file.

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
    o3d_mesh.compute_vertex_normals(True)
    mesh: TriangleMesh = TriangleMesh(
        np.asarray(o3d_mesh.vertices),
        np.asarray(o3d_mesh.vertex_colors),
        np.asarray(o3d_mesh.triangles),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals)
    )
    return MeshRenderer(mesh)


def render_window(*, drone_image: np.ndarray, drone_w_t_c: np.ndarray, image_renderer: OpenGLImageRenderer,
                  intrinsics: Tuple[float, float, float, float], scene_mesh_renderer: MeshRenderer,
                  tello_mesh_renderer: MeshRenderer, viewing_pose: np.ndarray, window_size: Tuple[int, int]) -> None:
    # Clear the window.
    OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), window_size)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render the whole scene from the viewing pose.
    OpenGLUtil.set_viewport((0.0, 0.0), (0.5, 1.0), window_size)

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

            # TODO
            scene_mesh_renderer.render()

            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(drone_w_t_c)):
                tello_mesh_renderer.render()

    glDisable(GL_DEPTH_TEST)

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

    # TODO
    scene_mesh_renderer: MeshRenderer = make_mesh_renderer(
        o3d.io.read_triangle_mesh("C:/spaint/build/bin/apps/spaintgui/meshes/groundtruth-decimated.ply")
    )

    tello_mesh_renderer: MeshRenderer = make_mesh_renderer(load_tello_mesh("C:/smglib/meshes/tello.ply"))

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.025
    )

    with SimulatedDrone(
        image_renderer=scene_mesh_renderer.render_to_image,
        image_size=(640, 480),
        intrinsics=intrinsics
    ) as drone:
        # Construct the image renderer.
        with OpenGLImageRenderer() as image_renderer:
            pygame.mixer.music.load("C:/smglib/sounds/drone_flying.mp3")

            # Stop when both Button 0 and Button 1 on the Futaba T6K are set to their "released" state.
            while joystick.get_button(0) != 0 or joystick.get_button(1) != 0:
                # Process any PyGame events.
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        # If Button 0 on the Futaba T6K is set to its "pressed" state, take off.
                        if event.button == 0:
                            if drone.get_state() == SimulatedDrone.IDLE:
                                pygame.mixer.music.play(loops=-1)
                            drone.takeoff()
                    elif event.type == pygame.JOYBUTTONUP:
                        # If Button 0 on the Futaba T6K is set to its "released" state, land.
                        if event.button == 0:
                            drone.land()
                    elif event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)

                # TODO
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

                # TODO
                drone_image, drone_camera_w_t_c, drone_w_t_c = drone.get_image_and_poses()
                print(drone_camera_w_t_c)

                # Allow the user to control the camera.
                camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                # Render the contents of the window.
                render_window(
                    drone_image=drone_image,
                    drone_w_t_c=drone_w_t_c,
                    image_renderer=image_renderer,
                    intrinsics=intrinsics,
                    scene_mesh_renderer=scene_mesh_renderer,
                    tello_mesh_renderer=tello_mesh_renderer,
                    viewing_pose=camera_controller.get_pose(),
                    window_size=window_size
                )


if __name__ == "__main__":
    main()
