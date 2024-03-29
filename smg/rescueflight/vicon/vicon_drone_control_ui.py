import cv2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import threading

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, List, Optional, Sequence, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.meshing import MeshUtil
from smg.navigation import OCS_OCCUPIED, PlanningToolkit
from smg.opengl import CameraRenderer, SceneRenderer
from smg.opengl import OpenGLImageRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.pyoctomap import CM_COLOR_HEIGHT, OctomapPicker, OctomapUtil, OcTree, OcTreeDrawer
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotorcontrol import DroneControllerFactory
from smg.rotorcontrol.controllers import DroneController
from smg.rotory.drones import Drone
from smg.utility import ImageUtil
from smg.vicon import LiveViconInterface


class ViconDroneControlUI:
    """A user interface (UI) for controlling a drone that uses a Vicon system to track the drone."""

    # CONSTRUCTOR

    def __init__(self, *, audio_input_device: Optional[int] = None, debug: bool = False, drone: Drone,
                 drone_controller_type: str, drone_mesh: o3d.geometry.TriangleMesh, drone_subject_name: str = "Tello",
                 intrinsics: Tuple[float, float, float, float], mapping_client: Optional[MappingClient],
                 planning_octree_filename: Optional[str], scene_mesh_filename: Optional[str],
                 scene_octree_filename: Optional[str], window_size: Tuple[int, int] = (1280, 480)):
        """
        Construct a drone UI that uses a Vicon system to track the drone.

        :param audio_input_device:          The index of the device to use for audio input (optional).
        :param debug:                       Whether to print out debugging messages.
        :param drone:                       The drone.
        :param drone_controller_type:       The type of drone controller to use.
        :param drone_mesh:                  An Open3D mesh for the drone.
        :param drone_subject_name:          The name of the Vicon subject corresponding to the drone.
        :param intrinsics:                  The camera intrinsics.
        :param mapping_client:              The mapping client (if any) to use to stream frames to a mapping server.
        :param planning_octree_filename:    The name of a file containing an octree for path planning (optional).
        :param scene_mesh_filename:         The name of a file containing a mesh for the scene (optional).
        :param scene_octree_filename:       The name of a file containing an octree for the scene (optional).
        :param window_size:                 The size of window to use.
        """
        self.__alive: bool = False

        self.__audio_input_device: Optional[int] = audio_input_device
        self.__debug: bool = debug
        self.__drone: Drone = drone
        self.__drone_controller: Optional[DroneController] = None
        self.__drone_controller_type: str = drone_controller_type
        self.__drone_mesh: Optional[OpenGLTriMesh] = None
        self.__drone_mesh_o3d: o3d.geometry.TriangleMesh = drone_mesh
        self.__drone_subject_name: str = drone_subject_name
        self.__frame_idx: int = 0
        self.__gl_image_renderer: Optional[OpenGLImageRenderer] = None
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__mapping_client: Optional[MappingClient] = mapping_client
        self.__octree_drawer: Optional[OcTreeDrawer] = None
        self.__planning_octree: Optional[OcTree] = None
        self.__planning_octree_filename: Optional[str] = planning_octree_filename
        self.__planning_toolkit: Optional[PlanningToolkit] = None
        self.__should_terminate: threading.Event = threading.Event()
        self.__scene_mesh: Optional[OpenGLTriMesh] = None
        self.__scene_mesh_filename: Optional[str] = scene_mesh_filename
        self.__scene_octree: Optional[OcTree] = None
        self.__scene_octree_filename: Optional[str] = scene_octree_filename
        self.__scene_octree_picker: Optional[OctomapPicker] = None
        self.__scene_renderer: Optional[SceneRenderer] = None
        self.__third_person: bool = False
        self.__vicon: LiveViconInterface = LiveViconInterface()
        self.__window_size: Tuple[int, int] = window_size

        self.__alive = True

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the drone UI's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the drone UI at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the drone UI."""
        # Initialise PyGame and some of its modules.
        pygame.init()
        pygame.joystick.init()
        pygame.mixer.init()

        # Make sure PyGame always gets the user inputs.
        pygame.event.set_grab(True)

        # Create the window.
        pygame.display.set_mode(self.__window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Vicon Drone Control UI")

        # Construct the OpenGL image renderer.
        self.__gl_image_renderer = OpenGLImageRenderer()

        # Construct the scene renderer.
        self.__scene_renderer = SceneRenderer()

        # Construct the octree drawer.
        self.__octree_drawer = OcTreeDrawer()
        self.__octree_drawer.set_color_mode(CM_COLOR_HEIGHT)

        # Convert the Open3D mesh for the drone to an OpenGL one so that it can be rendered.
        self.__drone_mesh = MeshUtil.convert_trimesh_to_opengl(self.__drone_mesh_o3d)
        self.__drone_mesh_o3d = None

        # Try to load in any octree that has been provided for path planning, and construct a planning toolkit
        # for it if it's available.
        if self.__planning_octree_filename is not None:
            self.__planning_octree = OctomapUtil.load_octree(self.__planning_octree_filename)
            if self.__planning_octree is not None:
                self.__planning_toolkit = PlanningToolkit(
                    self.__planning_octree,
                    neighbours=PlanningToolkit.neighbours6,
                    node_is_free=lambda n: self.__planning_toolkit.occupancy_status(n) != OCS_OCCUPIED
                )

        # Try to load in any mesh that has been provided for the scene.
        if self.__scene_mesh_filename is not None:
            self.__scene_mesh = MeshUtil.convert_trimesh_to_opengl(
                o3d.io.read_triangle_mesh(self.__scene_mesh_filename)
            )

        # Try to load in any octree that has been provided for the scene, and construct a picker for it if
        # it's available.
        width, height = self.__window_size
        if self.__scene_octree_filename is not None:
            self.__scene_octree = OctomapUtil.load_octree(self.__scene_octree_filename)
            if self.__scene_octree is not None:
                self.__scene_octree_picker = OctomapPicker(self.__scene_octree, width // 2, height, self.__intrinsics)

        # Get the drone's camera parameters.
        image_size: Tuple[int, int] = self.__drone.get_image_size()
        intrinsics: Optional[Tuple[float, float, float, float]] = self.__drone.get_intrinsics()
        assert (intrinsics is not None)

        # If we're using the mapping client, send a calibration message to tell the server the camera parameters.
        if self.__mapping_client is not None:
            self.__mapping_client.send_calibration_message(RGBDFrameMessageUtil.make_calibration_message(
                image_size, image_size, intrinsics, intrinsics
            ))

        # Construct the camera controller.
        camera_controller: KeyboardCameraController = KeyboardCameraController(
            CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.075
        )

        # Construct the drone controller.
        kwargs: Dict[str, dict] = {
            "aws_transcribe": dict(
                audio_input_device=self.__audio_input_device, debug=True, drone=self.__drone
            ),
            "futaba_t6k": dict(drone=self.__drone),
            "keyboard": dict(drone=self.__drone),
            "rts": dict(
                debug=False, drone=self.__drone, picker=self.__scene_octree_picker,
                planning_toolkit=self.__planning_toolkit, viewing_camera=camera_controller.get_camera()
            )
        }

        self.__drone_controller = DroneControllerFactory.make_drone_controller(
            self.__drone_controller_type, **kwargs[self.__drone_controller_type]
        )

        # Until the UI should terminate:
        while not self.__should_terminate.is_set():
            # Process any PyGame events.
            events: List[pygame.event.Event] = []
            for event in pygame.event.get():
                # Record the event for later use by the drone controller.
                events.append(event)

                if event.type == pygame.KEYDOWN:
                    # If the user presses the 't' key:
                    if event.key == pygame.K_t:
                        # Toggle the third-person view.
                        self.__third_person = not self.__third_person
                elif event.type == pygame.QUIT:
                    # If the user wants us to quit, first try to make sure the drone has landed, then exit.
                    drone_state: Optional[Drone.EState] = self.__drone.get_state()
                    if drone_state is not None and drone_state != Drone.IDLE:
                        self.__drone.stop()
                        self.__drone.land()

                    return

            # Also quit if the drone controller has finished.
            if self.__drone_controller.has_finished():
                return

            # Get the most recent image from the drone.
            drone_image: np.ndarray = self.__drone.get_image()

            # Try to get the drone's pose from the Vicon system.
            drone_w_t_c: Optional[np.ndarray] = None
            if self.__vicon.get_frame():
                drone_c_t_w: Optional[np.ndarray] = self.__vicon.get_segment_global_pose(
                    self.__drone_subject_name, self.__drone_subject_name
                )
                if drone_c_t_w is not None:
                    drone_w_t_c = np.linalg.inv(drone_c_t_w)
                    m: np.ndarray = np.array([
                        [1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]
                    ])
                    drone_w_t_c = m @ drone_w_t_c

            # If that succeeded:
            if drone_w_t_c is not None:
                # If we're using the mapping client, send the frame across to the server.
                if self.__mapping_client is not None:
                    dummy_depth_image: np.ndarray = np.zeros(drone_image.shape[:2], dtype=np.float32)
                    self.__mapping_client.send_frame_message(lambda msg: RGBDFrameMessageUtil.fill_frame_message(
                        self.__frame_idx, drone_image, ImageUtil.to_short_depth(dummy_depth_image),
                        drone_w_t_c, msg
                    ))

                # Increment the frame index.
                self.__frame_idx += 1

            # Allow the user to control the drone.
            self.__drone_controller.iterate(
                events=events, image=drone_image, intrinsics=self.__drone.get_intrinsics(),
                tracker_c_t_i=np.linalg.inv(drone_w_t_c) if drone_w_t_c is not None else None
            )

            # Get the keys that are currently being pressed by the user.
            pressed_keys: Sequence[bool] = pygame.key.get_pressed()

            # Move the free-view camera based on the keys that are being pressed.
            camera_controller.update(pressed_keys, timer() * 1000)

            # Render the contents of the window.
            self.__render_window(
                drone_image=drone_image,
                drone_w_t_c=drone_w_t_c,
                viewing_pose=camera_controller.get_pose()
            )

    def terminate(self) -> None:
        """Destroy the drone UI."""
        if self.__alive:
            # Set the termination flag if it isn't set already.
            if not self.__should_terminate.is_set():
                self.__should_terminate.set()

            # If the drone controller exists, destroy it.
            if self.__drone_controller is not None:
                self.__drone_controller.terminate()

            # If the scene renderer exists, destroy it.
            if self.__scene_renderer is not None:
                self.__scene_renderer.terminate()

            # If the OpenGL image renderer exists, destroy it.
            if self.__gl_image_renderer is not None:
                self.__gl_image_renderer.terminate()

            # If the Vicon interface is running, terminate it.
            if self.__vicon is not None:
                self.__vicon.terminate()

            # Shut down pygame and close any remaining OpenCV windows.
            pygame.quit()
            cv2.destroyAllWindows()

            self.__alive = False

    # PRIVATE METHODS

    def __render_window(self, *, drone_image: np.ndarray, drone_w_t_c: Optional[np.ndarray],
                        viewing_pose: np.ndarray) -> None:
        """
        Render the contents of the window.

        :param drone_image:     The most recent image from the drone.
        :param drone_w_t_c:     The pose of the drone (if known).
        :param viewing_pose:    The pose of the free-view camera.
        """
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

                # Render the mesh for the drone (at its current pose).
                if drone_w_t_c is not None:
                    with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(drone_w_t_c)):
                        SceneRenderer.render(lambda: self.__drone_mesh.render(), use_backface_culling=True)

                # Render the scene itself.
                if self.__scene_octree is not None:
                    SceneRenderer.render(
                        lambda: OctomapUtil.draw_octree_octovis(self.__scene_octree, self.__octree_drawer)
                    )
                elif self.__scene_mesh is not None:
                    SceneRenderer.render(self.__scene_mesh.render)

                # Render the UI for the drone controller.
                self.__drone_controller.render_ui()

        glPopAttrib()

        # Render the drone image.
        OpenGLUtil.set_viewport((0.5, 0.0), (1.0, 1.0), self.__window_size)
        self.__gl_image_renderer.render_image(ImageUtil.flip_channels(drone_image))

        # Swap the front and back buffers.
        pygame.display.flip()
