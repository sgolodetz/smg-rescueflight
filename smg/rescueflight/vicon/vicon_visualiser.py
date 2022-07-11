import bisect
import cv2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import threading
import time

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, Callable, List, Optional, Tuple

from smg.comms.base import RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.meshing import MeshUtil
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.skeletons import Skeleton3D, SkeletonRenderer
from smg.smplx import SMPLBody
from smg.utility import FiducialUtil, MarkerUtil, PoseUtil, SequenceUtil
from smg.vicon import LiveViconInterface, OfflineViconInterface, SubjectFromSourceCache
from smg.vicon import ViconFrameSaver, ViconInterface, ViconSkeletonDetector, ViconUtil


class ViconVisualiser:
    """A visualiser that supports the rendering, saving and replaying of Vicon-based scenes."""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, mapping_server: Optional[MappingServer],
                 pause: bool = False, persistence_folder: Optional[str], persistence_mode: str,
                 rendering_intrinsics: Tuple[float, float, float, float], use_partial_frames: bool,
                 use_vicon_poses: bool, window_size: Tuple[int, int] = (640, 480)):
        """
        Construct a Vicon visualiser.

        :param debug:                   Whether to enable debugging.
        :param mapping_server:          The mapping server (if any) that should be used to receive images from a drone.
        :param pause:                   Whether to start the visualiser in its paused state.
        :param persistence_folder:      The folder (if any) that should be used for Vicon persistence.
        :param persistence_mode:        The Vicon persistence mode.
        :param rendering_intrinsics:    The camera intrinsics to use when rendering the scene.
        :param use_partial_frames:      Whether to use the Vicon frames for which no corresponding image is available.
        :param use_vicon_poses:         Whether to use the joint poses produced by the Vicon system.
        :param window_size:             The application window size, as a (width, height) tuple.
        """
        self.__camera_controller: KeyboardCameraController = KeyboardCameraController(
            SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
        )
        self.__client_id: int = 0
        self.__debug: bool = debug
        self.__female_body: Optional[SMPLBody] = None
        self.__male_body: Optional[SMPLBody] = None
        self.__mapping_server: Optional[MappingServer] = mapping_server
        self.__pause: threading.Event = threading.Event()
        if pause:
            self.__pause.set()
        self.__persistence_folder: Optional[str] = persistence_folder
        self.__persistence_mode: str = persistence_mode
        self.__previous_frame_number: Optional[int] = None
        self.__previous_frame_start: Optional[float] = None
        self.__process_next: bool = True
        self.__receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        self.__rendering_intrinsics: Tuple[float, float, float, float] = rendering_intrinsics
        self.__scene_mesh: Optional[OpenGLTriMesh] = None
        self.__should_terminate: threading.Event = threading.Event()
        self.__skeleton_detector: Optional[ViconSkeletonDetector] = None
        self.__smpl_bodies: Dict[str, SMPLBody] = {}
        self.__subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")
        self.__subject_mesh_cache: Dict[str, OpenGLTriMesh] = {}
        self.__subject_mesh_loaders: Dict[str, Callable[[], OpenGLTriMesh]] = {
            "Tello": lambda: MeshUtil.convert_trimesh_to_opengl(MeshUtil.load_tello_mesh())
        }
        self.__use_partial_frames: bool = use_partial_frames
        self.__use_vicon_poses: bool = use_vicon_poses
        self.__vicon: Optional[ViconInterface] = None
        self.__vicon_frame_numbers: List[int] = []
        self.__vicon_frame_saver: Optional[ViconFrameSaver] = None
        self.__vicon_frame_timestamps: List[float] = []
        self.__window_size: Tuple[int, int] = window_size

        # Construct the lock.
        self.__lock: threading.Lock = threading.Lock()

        # If we're running a mapping server, start the image processing thread.
        self.__image_processing_thread: Optional[threading.Thread] = None
        if self.__mapping_server is not None:
            self.__image_processing_thread = threading.Thread(target=self.__process_images)
            self.__image_processing_thread.start()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the visualiser's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the visualiser at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the visualiser."""
        # Initialise PyGame and create the window.
        pygame.init()
        pygame.display.set_mode(self.__window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Vicon Visualiser")

        # Construct the Vicon interface.
        if self.__persistence_mode == "input":
            self.__vicon = OfflineViconInterface(
                folder=self.__persistence_folder, use_partial_frames=self.__use_partial_frames
            )
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
        self.__female_body = SMPLBody("female", texture_image_filename="surreal/nongrey_female_0891.jpg")
        self.__male_body = SMPLBody("male", texture_image_filename="surreal/nongrey_male_0170.jpg")

        self.__smpl_bodies["Aluna"] = self.__female_body
        self.__smpl_bodies["Madhu"] = self.__male_body

        # If we're in input mode, load in a ground-truth scene mesh (if available).
        if self.__persistence_mode == "input" and self.__vicon.get_frame():
            self.__scene_mesh = self.__try_load_scene_mesh(self.__vicon)

        # Until the visualiser should terminate:
        while not self.__should_terminate.is_set():
            # Process any PyGame events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # If the user presses the 'b' key, process frames without pausing.
                    if event.key == pygame.K_b:
                        self.__pause.clear()
                        self.__process_next = True

                    # Otherwise, if the user presses the 'n' key, process the next frame and then pause.
                    elif event.key == pygame.K_n:
                        self.__pause.set()
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
                self.__process_next = not self.__pause.is_set()

            # Print out the frame number.
            print(f"=== Frame {self.__vicon.get_frame_number()} ===")

            # Allow the user to control the camera.
            self.__camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Render the frame.
            self.__render_frame()

    def terminate(self) -> None:
        """Destroy the visualiser."""
        # If the termination flag hasn't previously been set:
        if not self.__should_terminate.is_set():
            # Set it now.
            self.__should_terminate.set()

            # Wait for the image processing thread to terminate.
            if self.__image_processing_thread is not None:
                self.__image_processing_thread.join()

            # If the Vicon interface is running, terminate it.
            if self.__vicon is not None:
                self.__vicon.terminate()

    # PRIVATE METHODS

    def __advance_to_next_frame(self) -> None:
        """Try to advance to the next frame."""
        # Try to get a frame from the Vicon system. If that succeeds:
        if self.__vicon.get_frame():
            # Get the frame number and calculate a timestamp for the frame, and record both for later.
            frame_number: int = self.__vicon.get_frame_number()
            frame_timestamp: float = time.time_ns() / 1000
            with self.__lock:
                self.__vicon_frame_numbers.append(frame_number)
                self.__vicon_frame_timestamps.append(frame_timestamp)

            # If we're in output mode, save the frame to disk.
            if self.__persistence_mode == "output":
                self.__vicon_frame_saver.save_frame()

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

    def __process_images(self) -> None:
        """Process images received from the mapping client."""
        # While the visualiser should not terminate:
        while not self.__should_terminate.is_set():
            # If we're not paused and a frame is available from the client:
            if not self.__pause.is_set() and self.__mapping_server.has_frames_now(self.__client_id):
                # Get the camera parameters from the server.
                intrinsics: Optional[Tuple[float, float, float, float]] = self.__mapping_server.get_intrinsics(
                    self.__client_id
                )[0]

                # Get the oldest frame from the client that hasn't yet been processed.
                self.__mapping_server.get_frame(self.__client_id, self.__receiver)
                colour_image: np.ndarray = self.__receiver.get_rgb_image()
                image_timestamp: Optional[float] = self.__receiver.get_frame_timestamp()
                pose: np.ndarray = self.__receiver.get_pose()

                # If we're debugging, show the received colour image:
                if self.__debug:
                    cv2.imshow("Received Image", colour_image)
                    cv2.waitKey(1)

                # If we're in output mode:
                if self.__persistence_mode == "output":
                    # If the camera parameters haven't already been saved to disk, save them now.
                    calibration_filename: str = SequenceUtil.make_calibration_filename(self.__persistence_folder)
                    if not os.path.exists(calibration_filename):
                        image_size: Tuple[int, int] = (colour_image.shape[1], colour_image.shape[0])
                        SequenceUtil.save_rgbd_calibration(
                            self.__persistence_folder, image_size, image_size, intrinsics, intrinsics
                        )

                    # If the image from the mapping client has a valid timestamp:
                    if image_timestamp is not None:
                        # Find the Vicon frame(s) with the closest timestamps to that of the image (the 'candidates'),
                        # and calculate the differences (the 'deltas') between their timestamps and that of the image.
                        with self.__lock:
                            i: int = bisect.bisect_left(self.__vicon_frame_timestamps, image_timestamp)
                            candidates: List[int] = []
                            deltas: List[float] = []
                            for j in [i-1, i]:
                                if 0 <= j < len(self.__vicon_frame_timestamps):
                                    candidates.append(self.__vicon_frame_numbers[j])
                                    deltas.append(abs(self.__vicon_frame_timestamps[j] - image_timestamp))

                        # Find the best candidate, i.e. the one whose timestamp is closest to that of the image.
                        best_candidate_idx: int = np.argmin(deltas)

                        # If the difference between the timestamp of the best candidate and that of the image is
                        # no greater than the specified threshold:
                        # TODO: Set an appropriate threshold here.
                        if True:  # deltas[k] <= some threshold
                            # Get the frame number of the best candidate.
                            frame_number: int = candidates[best_candidate_idx]

                            # Save the frame to disk.
                            colour_filename: str = os.path.join(
                                self.__persistence_folder, f"{frame_number}.color.png"
                            )
                            pose_filename: str = os.path.join(
                                self.__persistence_folder, f"{frame_number}.pose.txt"
                            )
                            cv2.imwrite(colour_filename, colour_image)
                            PoseUtil.save_pose(pose_filename, pose)

            # Otherwise:
            else:
                # Wait for 10 milliseconds to avoid a spin loop.
                time.sleep(0.01)

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
                OpenGLUtil.render_voxel_grid([-3, -7, 0], [3, 7, 2], [1, 1, 1], dotted=True)

                # Render the scene mesh (if any).
                if self.__scene_mesh is not None:
                    self.__scene_mesh.render()

                # Detect any skeletons in the frame.
                skeletons: Dict[str, Skeleton3D] = self.__skeleton_detector.detect_skeletons()

                # Compute the subject designations for the frame. If we're debugging, also print them out.
                subject_designations: Dict[str, List[Tuple[str, float]]] = ViconUtil.compute_subject_designations(
                    self.__vicon, skeletons
                )

                if self.__debug:
                    print(subject_designations)

                # For each Vicon subject:
                for subject in self.__vicon.get_subject_names():
                    # Render all of its markers.
                    with ViconUtil.default_lighting_context():
                        for marker_name, marker_pos in self.__vicon.get_marker_positions(subject).items():
                            glColor3f(1.0, 0.0, 0.0)
                            OpenGLUtil.render_sphere(marker_pos, 0.014, slices=10, stacks=10)

                    # If the subject is a person, try to render its skeleton and SMPL body (if available).
                    if ViconUtil.is_person(subject, self.__vicon):
                        self.__render_person(subject, skeletons)
                    else:
                        # Otherwise, hypothesise that it's a single-segment subject and try to get its pose.
                        subject_from_vicon: Optional[np.ndarray] = self.__vicon.get_segment_global_pose(
                            subject, subject
                        )

                        # If that succeeds, render the subject based on its type:
                        if subject_from_vicon is not None:
                            subject_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(subject_from_vicon)
                            subject_from_source: Optional[np.ndarray] = self.__subject_from_source_cache.get(subject)

                            if subject_from_source is not None:
                                source_from_vicon: np.ndarray = np.linalg.inv(subject_from_source) @ subject_from_vicon
                                self.__render_image_source(subject, source_from_vicon)
                            elif ViconUtil.is_designatable(subject):
                                self.__render_designatable_subject(subject, subject_cam.p(), subject_designations)
                            else:
                                # Treat the subject as a generic one and simply render its Vicon-obtained pose.
                                CameraRenderer.render_camera(subject_cam, axis_scale=0.5)

                # Disable the z-buffer again.
                glPopAttrib()

        # Swap the front and back buffers.
        pygame.display.flip()

    def __render_image_source(self, subject: str, source_from_vicon: np.ndarray) -> None:
        """
        Render an image source (e.g. a drone).

        :param subject:             The Vicon subject corresponding to the image source.
        :param source_from_vicon:   A transformation from Vicon space to the space of the image source.
        """
        # Render the coordinate axes of the image source.
        source_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(source_from_vicon)
        glLineWidth(5)
        CameraRenderer.render_camera(source_cam, axis_scale=0.5)
        glLineWidth(1)

        # If a mesh for the image source is available, render it.
        subject_mesh: Optional[OpenGLTriMesh] = self.__try_get_subject_mesh(subject)
        if subject_mesh is not None:
            vicon_from_source: np.ndarray = np.linalg.inv(source_from_vicon)
            with ViconUtil.default_lighting_context():
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(vicon_from_source)):
                    subject_mesh.render()

    def __render_person(self, subject: str, skeletons: Dict[str, Skeleton3D]) -> None:
        """
        Render a person.

        :param subject:     The name of the Vicon subject corresponding to the person.
        :param skeletons:   The skeletons that have been detected in the frame.
        """
        # If a skeleton was detected for the person, render both that and the corresponding SMPL body.
        skeleton: Optional[Skeleton3D] = skeletons.get(subject)
        if skeleton is not None:
            with ViconUtil.default_lighting_context():
                SkeletonRenderer.render_skeleton(skeleton)

                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                body: SMPLBody = self.__smpl_bodies.get(subject, self.__male_body)
                body.render_from_skeleton(skeleton)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            SkeletonRenderer.render_keypoint_poses(skeleton)

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

    def __try_load_scene_mesh(self, vicon: ViconInterface) -> Optional[OpenGLTriMesh]:
        """
        Try to load in a ground-truth scene mesh, ensuring it is in the Vicon coordinate system in the process.

        :param vicon:   The Vicon interface.
        :return:        The ground-truth scene mesh.
        """
        gt_folder: str = os.path.join(self.__persistence_folder, "gt")
        scene_mesh_o3d: Optional[o3d.geometry.TriangleMesh] = None

        # If there's a copy of the ground-truth mesh that's already in Vicon space:
        vicon_mesh_filename: str = os.path.join(gt_folder, "vicon_mesh.ply")
        if os.path.exists(vicon_mesh_filename):
            # Load that in and return it.
            scene_mesh_o3d = o3d.io.read_triangle_mesh(vicon_mesh_filename)

        # Otherwise:
        else:
            # If the ground-truth mesh reconstructed by SemanticPaint and the required fiducials are available:
            mesh_filename: str = os.path.join(gt_folder, "mesh.ply")
            fiducials_filename: str = os.path.join(gt_folder, "fiducials.txt")
            if os.path.exists(mesh_filename) and os.path.exists(fiducials_filename):
                # Load in the positions of the four ArUco marker corners as estimated by SemanticPaint.
                gt_marker_positions: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(fiducials_filename)

                # Look up the positions of all of the Vicon markers for the ArUco marker's Vicon subject
                # that can currently be seen by the Vicon system, hopefully including all of the ones
                # for the ArUco marker's corners.
                vicon_marker_positions: Dict[str, np.ndarray] = vicon.get_marker_positions("Registrar")

                # Try to estimate the rigid transformation from ground-truth space to Vicon space.
                vicon_from_gt: Optional[np.ndarray] = MarkerUtil.estimate_space_to_space_transform(
                    gt_marker_positions, vicon_marker_positions
                )

                # If that succeeds, load in the ground-truth scene mesh and transform it into Vicon space.
                if vicon_from_gt is not None:
                    scene_mesh_o3d = o3d.io.read_triangle_mesh(mesh_filename)
                    scene_mesh_o3d.transform(vicon_from_gt)

        # If a scene mesh was loaded, convert it to OpenGL format and return it; otherwise, return None.
        return MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d) if scene_mesh_o3d is not None else None

    # PRIVATE STATIC METHODS

    @staticmethod
    def __render_designatable_subject(subject_name: str, subject_pos: np.ndarray,
                                      designations: Dict[str, List[Tuple[str, float]]]) -> None:
        """
        Render a designatable subject.

        .. note::
            The subject will be rendered as a coloured sphere, where the colour depends on the extent to which
            the subject is currently being designated by any detected skeleton in the frame. The way in which
            designation is specified can be found in ViconUtil.compute_subject_designations.

        :param subject_name:    The name of the subject.
        :param subject_pos:     The position of the subject (i.e. the origin of its coordinate system).
        :param designations:    The subject designations for the frame.
        """
        colour: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        designations_for_subject: Optional[List[Tuple[str, float]]] = designations.get(subject_name)
        if designations_for_subject is not None:
            # Note: The designations for each subject are sorted in non-decreasing order of distance.
            _, min_dist = designations_for_subject[0]
            t: float = np.clip(min_dist / 0.5, 0.0, 1.0)
            colour = (t, 1 - t, 0)

        with ViconUtil.default_lighting_context():
            glColor3f(*colour)
            OpenGLUtil.render_sphere(subject_pos, 0.1, slices=10, stacks=10)
