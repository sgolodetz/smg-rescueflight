import math
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import threading
import time

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Optional, Tuple

from smg.navigation import AStarPathPlanner, OCS_OCCUPIED, PlanningToolkit
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLSceneRenderer, OpenGLTriMesh, OpenGLUtil
from smg.pyoctomap import OcTree
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil


class DroneSimulator:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, intrinsics: Tuple[float, float, float, float], plan_paths: bool = False,
                 tello_mesh_filename: str, window_size: Tuple[int, int] = (1280, 480)):
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__plan_paths: bool = plan_paths
        self.__should_terminate: threading.Event = threading.Event()
        self.__tello_mesh_filename: str = tello_mesh_filename
        self.__window_size: Tuple[int, int] = window_size

        # The threads and conditions.
        self.__planning_thread: Optional[threading.Thread] = None

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the simulator's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the simulator at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the drone simulator."""
        # Initialise pygame and some of its modules.
        pygame.init()
        pygame.joystick.init()
        pygame.mixer.init()

        # Make sure pygame always gets the user inputs.
        pygame.event.set_grab(True)

        # Create the window.
        pygame.display.set_mode(self.__window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Drone Simulator")

        # Load in the mesh for the drone, and prepare it for rendering.
        drone_mesh: OpenGLTriMesh = DroneSimulator.__convert_trimesh_to_opengl(
            DroneSimulator.__load_tello_mesh(self.__tello_mesh_filename)
        )

        # Construct the camera controller.
        camera_controller: KeyboardCameraController = KeyboardCameraController(
            CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.025
        )

        # If we're planning paths, start the path planning thread.
        if self.__plan_paths:
            self.__planning_thread = threading.Thread(target=self.__run_planning)
            self.__planning_thread.start()

        # Until the simulator should terminate:
        while not self.__should_terminate.is_set():
            # Process any PyGame events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # If the user wants us to quit, do so.
                    pygame.quit()
                    return

            # Get the drone's image and poses.
            # drone_image, drone_camera_w_t_c, drone_chassis_w_t_c = self.__drone.get_image_and_poses()

            # Allow the user to control the free-view camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Render the contents of the window.
            self.__render_window(
                drone_chassis_w_t_c=np.eye(4),  # drone_chassis_w_t_c,
                drone_mesh=drone_mesh,
                viewing_pose=camera_controller.get_pose()
            )

    def terminate(self) -> None:
        """Destroy the simulator."""
        if not self.__should_terminate.is_set():
            self.__should_terminate.set()

            # Join any running threads.
            if self.__planning_thread is not None:
                self.__planning_thread.join()

    # PRIVATE METHODS

    def __render_window(self, *, drone_chassis_w_t_c: np.ndarray, drone_mesh: OpenGLTriMesh, viewing_pose: np.ndarray) \
            -> None:
        """Render the contents of the window."""
        # Clear the window.
        OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), self.__window_size)
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render the whole scene from the viewing pose.
        OpenGLUtil.set_viewport((0.0, 0.0), (0.5, 1.0), self.__window_size)

        glPushAttrib(GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST)

        width, height = self.__window_size
        with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
            self.__intrinsics, width // 2, height
        )):
            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                CameraPoseConverter.pose_to_modelview(viewing_pose)
            )):
                # Render a voxel grid.
                glColor3f(0.0, 0.0, 0.0)
                OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                # Render coordinate axes.
                CameraRenderer.render_camera(CameraUtil.make_default_camera())

                # If possible, render the scene.
                # if render_scene is not None:
                #     render_scene()

                # Render the mesh for the drone (at its current pose).
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(drone_chassis_w_t_c)):
                    OpenGLSceneRenderer.render(lambda: drone_mesh.render())

        glPopAttrib()

        # Render the drone image.
        OpenGLUtil.set_viewport((0.5, 0.0), (1.0, 1.0), self.__window_size)
        # image_renderer.render_image(ImageUtil.flip_channels(drone_image))

        # Swap the front and back buffers.
        pygame.display.flip()

    def __run_planning(self) -> None:
        """Run the path planning thread."""
        # Load the planning octree.
        voxel_size: float = 0.1
        tree: OcTree = OcTree(voxel_size)
        tree.read_binary("C:/smglib/smg-mapping/output-navigation/octree10cm.bt")

        # Construct the planning toolkit.
        planning_toolkit: PlanningToolkit = PlanningToolkit(
            tree,
            neighbours=PlanningToolkit.neighbours6,
            node_is_free=lambda n: planning_toolkit.occupancy_status(n) != OCS_OCCUPIED
        )

        # Construct the path planner.
        planner: AStarPathPlanner = AStarPathPlanner(planning_toolkit, debug=True)

        # TODO
        while not self.__should_terminate.is_set():
            # TODO
            time.sleep(0.01)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __convert_trimesh_to_opengl(o3d_mesh: o3d.geometry.TriangleMesh) -> OpenGLTriMesh:
        """
        Convert an Open3D triangle mesh to an OpenGL one.

        :param o3d_mesh:    The Open3D triangle mesh.
        :return:            The OpenGL mesh.
        """
        # FIXME: This should probably be moved somewhere more central at some point.
        o3d_mesh.compute_vertex_normals(True)
        return OpenGLTriMesh(
            np.asarray(o3d_mesh.vertices),
            np.asarray(o3d_mesh.vertex_colors),
            np.asarray(o3d_mesh.triangles),
            vertex_normals=np.asarray(o3d_mesh.vertex_normals)
        )

    # noinspection PyArgumentList
    @staticmethod
    def __load_tello_mesh(filename: str) -> o3d.geometry.TriangleMesh:
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
