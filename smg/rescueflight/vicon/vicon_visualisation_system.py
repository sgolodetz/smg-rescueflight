import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import threading
import time

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, Callable, Optional, Tuple

from smg.comms.base import RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.meshing import MeshUtil
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.skeletons import Skeleton3D, SkeletonRenderer
from smg.smplx import SMPLBody
from smg.vicon import LiveViconInterface, OfflineViconInterface, SubjectFromSourceCache
from smg.vicon import ViconFrameSaver, ViconInterface, ViconSkeletonDetector, ViconUtil


class ViconVisualisationSystem:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, mapping_server: Optional[MappingServer],
                 pause: bool = False, persistence_folder: Optional[str], persistence_mode: str,
                 rendering_intrinsics: Tuple[float, float, float, float], scene_timestamp: Optional[str],
                 scenes_folder: str, use_vicon_poses: bool, window_size: Tuple[int, int] = (640, 480)):
        """
        TODO

        :param debug:                   TODO
        :param mapping_server:          TODO
        :param pause:                   TODO
        :param persistence_folder:      TODO
        :param persistence_mode:        TODO
        :param rendering_intrinsics:    TODO
        :param scene_timestamp:         TODO
        :param scenes_folder:           TODO
        :param use_vicon_poses:         TODO
        :param window_size:             TODO
        """
        self.__camera_controller: KeyboardCameraController = KeyboardCameraController(
            SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
        )
        self.__client_id: int = 0
        self.__debug: bool = debug
        self.__female_body: Optional[SMPLBody] = None
        self.__male_body: Optional[SMPLBody] = None
        self.__mapping_server: Optional[MappingServer] = mapping_server
        self.__pause: bool = pause
        self.__persistence_folder: Optional[str] = persistence_folder
        self.__persistence_mode: str = persistence_mode
        self.__previous_frame_number: Optional[int] = None
        self.__previous_frame_start: Optional[float] = None
        self.__process_next: bool = True
        self.__receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        self.__rendering_intrinsics: Tuple[float, float, float, float] = rendering_intrinsics
        self.__scene_mesh: Optional[OpenGLTriMesh] = None
        self.__scene_timestamp: Optional[str] = scene_timestamp
        self.__scenes_folder: str = scenes_folder
        self.__should_terminate: threading.Event = threading.Event()
        self.__skeleton_detector: Optional[ViconSkeletonDetector] = None
        self.__smpl_bodies: Dict[str, SMPLBody] = {}
        self.__subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")
        self.__subject_mesh_cache: Dict[str, OpenGLTriMesh] = {}
        self.__subject_mesh_loaders: Dict[str, Callable[[], OpenGLTriMesh]] = {
            "Tello": lambda: MeshUtil.convert_trimesh_to_opengl(MeshUtil.load_tello_mesh())
        }
        self.__use_vicon_poses: bool = use_vicon_poses
        self.__vicon: Optional[ViconInterface] = None
        self.__vicon_frame_saver: Optional[ViconFrameSaver] = None
        self.__window_size: Tuple[int, int] = window_size

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the visualisation system's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the visualisation system at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the visualisation system."""
        # Initialise PyGame and create the window.
        pygame.init()
        pygame.display.set_mode(self.__window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Vicon Visualisation System")

        # Construct the Vicon interface.
        if self.__persistence_mode == "input":
            self.__vicon = OfflineViconInterface(folder=self.__persistence_folder)
        else:
            self.__vicon = LiveViconInterface()

        # Construct the skeleton detector.
        self.__skeleton_detector = ViconSkeletonDetector(
            self.__vicon, is_person=ViconUtil.is_person, use_vicon_poses=self.__use_vicon_poses
        )

        # If we're in output mode, construct the Vicon frame saver.
        if self.__persistence_mode == "output":
            self.__vicon_frame_saver = ViconFrameSaver(folder=self.__persistence_folder, vicon=self.__vicon)

        # Load the SMPL body models.
        self.__female_body = SMPLBody(
            "female",
            texture_coords_filename="D:/smplx/textures/smpl/texture_coords.npy",
            texture_image_filename="D:/smplx/textures/smpl/surreal/nongrey_female_0891.jpg"
        )

        self.__male_body = SMPLBody(
            "male",
            texture_coords_filename="D:/smplx/textures/smpl/texture_coords.npy",
            texture_image_filename="D:/smplx/textures/smpl/surreal/nongrey_male_0170.jpg"
        )

        self.__smpl_bodies["Aluna"] = self.__female_body
        self.__smpl_bodies["Madhu"] = self.__male_body

        # Load in the scene mesh (if any), transforming it as needed in the process.
        if self.__scene_timestamp is not None and self.__vicon.get_frame():
            self.__scene_mesh = ViconUtil.load_scene_mesh(self.__scenes_folder, self.__scene_timestamp, self.__vicon)

        # Until the visualisation system should terminate:
        while not self.__should_terminate.is_set():
            # Process any PyGame events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # If the user presses the 'b' key, process frames without pausing.
                    if event.key == pygame.K_b:
                        self.__pause = False
                        self.__process_next = True

                    # Otherwise, if the user presses the 'n' key, process the next frame and then pause.
                    elif event.key == pygame.K_n:
                        self.__pause = True
                        self.__process_next = True
                elif event.type == pygame.QUIT:
                    # If the user wants us to quit, shut down pygame, close any remaining OpenCV windows, and exit.
                    pygame.quit()
                    cv2.destroyAllWindows()
                    return

            # If we're ready to process the next frame:
            if self.__process_next:
                # Process the frame.
                self.__advance_to_next_frame()

                # Decide whether to continue processing subsequent frames or wait.
                self.__process_next = not self.__pause

            # Print out the frame number.
            print(f"=== Frame {self.__vicon.get_frame_number()} ===")

            # Allow the user to control the camera.
            self.__camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Render the frame.
            self.__render_frame()

    def terminate(self) -> None:
        """Destroy the visualisation system."""
        if not self.__should_terminate.is_set():
            self.__should_terminate.set()

            # If the Vicon interface is running, terminate it.
            if self.__vicon is not None:
                self.__vicon.terminate()

    # PRIVATE METHODS

    def __advance_to_next_frame(self) -> None:
        """Try to advance to the next frame."""
        # Try to get a frame from the Vicon system. If that succeeds:
        if self.__vicon.get_frame():
            frame_number: int = self.__vicon.get_frame_number()

            # If we're running a mapping server, try to get a frame from the client.
            colour_image: Optional[np.ndarray] = None
            if self.__mapping_server is not None and self.__mapping_server.has_frames_now(self.__client_id):
                # # Get the camera parameters from the server.
                # height, width, _ = self.__mapping_server.get_image_shapes(client_id)[0]
                # image_size = (width, height)
                # intrinsics = self.__mapping_server.get_intrinsics(client_id)[0]

                # Get the newest frame from the mapping server.
                self.__mapping_server.peek_newest_frame(self.__client_id, self.__receiver)
                colour_image = self.__receiver.get_rgb_image()

                # If we're debugging, show the received colour image:
                if self.__debug:
                    cv2.imshow("Received Image", colour_image)
                    cv2.waitKey(1)

            # If we're in output mode:
            if True:  # self.__persistence_mode == "output":
                # If we aren't running a mapping server:
                if self.__mapping_server is None:
                    # Save the Vicon frame to disk.
                    # self.__vicon_frame_saver.save_frame()
                    print("Would save Vicon frame")

                # Otherwise, if we are running a server and an image has been obtained from the client:
                elif colour_image is not None:
                    # Save the Vicon frame to disk.
                    # self.__vicon_frame_saver.save_frame()
                    print("Would save Vicon frame")

                    # Save the colour image to disk.
                    filename: str = os.path.join(self.__persistence_folder, f"{frame_number}.png")
                    print(f"Would save image to {filename}")

            # Check how long has elapsed since the start of the previous frame. If it's not long
            # enough, pause until the expected amount of time has elapsed.
            frame_start: float = timer()
            if self.__previous_frame_number is not None:
                recording_fps: int = 200
                expected_time_delta: float = (frame_number - self.__previous_frame_number) / recording_fps
                time_delta: float = frame_start - self.__previous_frame_start
                time_delta_offset: float = expected_time_delta - time_delta
                if time_delta_offset > 0:
                    time.sleep(time_delta_offset)

            self.__previous_frame_number = frame_number
            self.__previous_frame_start = frame_start

    def __render_frame(self) -> None:
        """Render the current frame."""
        # Clear the colour and depth buffers.
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix.
        with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
            self.__rendering_intrinsics, *self.__window_size
        )):
            # Set the model-view matrix.
            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                CameraPoseConverter.pose_to_modelview(self.__camera_controller.get_pose())
            )):
                # Enable the z-buffer.
                glPushAttrib(GL_DEPTH_BUFFER_BIT)
                glEnable(GL_DEPTH_TEST)
                glDepthFunc(GL_LESS)

                # Render coordinate axes.
                CameraRenderer.render_camera(
                    CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
                )

                # Render a voxel grid.
                glColor3f(0.0, 0.0, 0.0)
                OpenGLUtil.render_voxel_grid([-3, -5, 0], [3, 5, 2], [1, 1, 1], dotted=True)

                # Render the scene mesh (if any).
                if self.__scene_mesh is not None:
                    self.__scene_mesh.render()

                # For each Vicon subject:
                for subject in self.__vicon.get_subject_names():
                    # Render all of its markers.
                    for marker_name, marker_pos in self.__vicon.get_marker_positions(subject).items():
                        glColor3f(1.0, 0.0, 0.0)
                        OpenGLUtil.render_sphere(marker_pos, 0.014, slices=10, stacks=10)

                    # If the subject is a person, don't bother trying to render its (rigid-body) pose.
                    if ViconUtil.is_person(subject, self.__vicon):
                        continue

                    # Otherwise, assume it's a single-segment subject and try to get its pose.
                    subject_from_world: Optional[np.ndarray] = self.__vicon.get_segment_global_pose(subject, subject)

                    # If that succeeds:
                    if subject_from_world is not None:
                        # Assume that the subject corresponds to an image source, and try to get the
                        # relative transformation from that image source to the subject.
                        subject_from_source: Optional[np.ndarray] = self.__subject_from_source_cache.get(subject)

                        # If that succeeds (i.e. the subject does correspond to an image source, and we know the
                        # relative transformation):
                        if subject_from_source is not None:
                            # Render the pose of the image source.
                            source_from_world: np.ndarray = np.linalg.inv(subject_from_source) @ subject_from_world
                            source_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(source_from_world)
                            glLineWidth(5)
                            CameraRenderer.render_camera(source_cam, axis_scale=0.5)
                            glLineWidth(1)

                            # If the mesh for the subject is available, render it.
                            subject_mesh: Optional[OpenGLTriMesh] = self.__try_get_subject_mesh(subject)
                            if subject_mesh is not None:
                                world_from_source: np.ndarray = np.linalg.inv(source_from_world)
                                with ViconUtil.default_lighting_context():
                                    with OpenGLMatrixContext(
                                        GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(world_from_source)
                                    ):
                                        subject_mesh.render()

                        # Otherwise, if the subject doesn't correspond to an image source, or we don't know the
                        # relative transformation:
                        else:
                            # Render the subject pose obtained from the Vicon system.
                            subject_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(subject_from_world)
                            CameraRenderer.render_camera(subject_cam, axis_scale=0.5)

                # Detect any skeletons in the frame.
                skeletons: Dict[str, Skeleton3D] = self.__skeleton_detector.detect_skeletons()

                # Render the skeletons and their corresponding SMPL bodies.
                for subject, skeleton in skeletons.items():
                    with ViconUtil.default_lighting_context():
                        SkeletonRenderer.render_skeleton(skeleton)

                        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                        body: SMPLBody = self.__smpl_bodies.get(subject, self.__male_body)
                        body.render_from_skeleton(skeleton)
                        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                    SkeletonRenderer.render_keypoint_poses(skeleton)

                # Disable the z-buffer again.
                glPopAttrib()

        # Swap the front and back buffers.
        pygame.display.flip()

    def __try_get_subject_mesh(self, subject: str) -> Optional[OpenGLTriMesh]:
        """
        Try to get the mesh for a subject.

        :param subject: The name of the subject.
        :return:        The mesh for the subject, if possible, or None otherwise.
        """
        # Try to look up the mesh for the subject in the cache.
        subject_mesh: Optional[OpenGLTriMesh] = self.__subject_mesh_cache.get(subject)

        # If it's not there, try to load it into the cache.
        if subject_mesh is None:
            subject_mesh_loader: Optional[Callable[[], OpenGLTriMesh]] = self.__subject_mesh_loaders.get(subject)
            if subject_mesh_loader is not None:
                subject_mesh = subject_mesh_loader()
                self.__subject_mesh_cache[subject] = subject_mesh

        return subject_mesh
