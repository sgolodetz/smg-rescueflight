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
from smg.utility import FiducialUtil, GeometryUtil, PoseUtil, SequenceUtil
from smg.vicon import LiveViconInterface, OfflineViconInterface, SubjectFromSourceCache
from smg.vicon import ViconFrameSaver, ViconInterface, ViconSkeletonDetector, ViconUtil


class ViconVisualiser:
    """A visualiser that supports the rendering, saving and replaying of Vicon-based scenes."""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, mapping_server: Optional[MappingServer],
                 pause: bool = False, persistence_folder: Optional[str], persistence_mode: str,
                 rendering_intrinsics: Tuple[float, float, float, float], scene_timestamp: Optional[str],
                 scenes_folder: str, use_vicon_poses: bool, window_size: Tuple[int, int] = (640, 480)):
        """
        Construct a Vicon visualiser.

        :param debug:                   Whether to enable debugging.
        :param mapping_server:          The mapping server (if any) that should be used to receive images from a drone.
        :param pause:                   Whether to start the visualiser in its paused state.
        :param persistence_folder:      The folder (if any) that should be used for Vicon persistence.
        :param persistence_mode:        The Vicon persistence mode.
        :param rendering_intrinsics:    The camera intrinsics to use when rendering the scene.
        :param scene_timestamp:         A timestamp indicating which scene mesh to load (if any).
        :param scenes_folder:           The folder from which to load the scene mesh (if any).
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
            self.__scene_mesh = ViconVisualiser.__load_scene_mesh(
                self.__scenes_folder, self.__scene_timestamp, self.__vicon
            )

        # Until the visualiser should terminate:
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
        """Destroy the visualiser."""
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
            intrinsics: Optional[Tuple[float, float, float, float]] = None
            pose: Optional[np.ndarray] = None

            if self.__mapping_server is not None and self.__mapping_server.has_frames_now(self.__client_id):
                # Get the camera parameters from the server.
                intrinsics = self.__mapping_server.get_intrinsics(self.__client_id)[0]

                # Get the newest frame from the mapping server.
                self.__mapping_server.peek_newest_frame(self.__client_id, self.__receiver)
                colour_image = self.__receiver.get_rgb_image()
                pose = self.__receiver.get_pose()

                # If we're debugging, show the received colour image:
                if self.__debug:
                    cv2.imshow("Received Image", colour_image)
                    cv2.waitKey(1)

            # If we're in output mode:
            if self.__persistence_mode == "output":
                # If we aren't running a mapping server:
                if self.__mapping_server is None:
                    # Save the Vicon frame to disk.
                    self.__vicon_frame_saver.save_frame()

                # Otherwise, if we are running a server and an image has been obtained from the client:
                elif colour_image is not None:
                    # Save the Vicon frame to disk.
                    self.__vicon_frame_saver.save_frame()

                    # If the camera parameters haven't already been saved to disk, save them now.
                    calibration_filename: str = SequenceUtil.make_calibration_filename(self.__persistence_folder)
                    if not os.path.exists(calibration_filename):
                        image_size: Tuple[int, int] = (colour_image.shape[1], colour_image.shape[0])
                        SequenceUtil.save_rgbd_calibration(
                            self.__persistence_folder, image_size, image_size, intrinsics, intrinsics
                        )

                    # Save the frame from the mapping server to disk.
                    colour_filename: str = os.path.join(self.__persistence_folder, f"{frame_number}.color.png")
                    pose_filename: str = os.path.join(self.__persistence_folder, f"{frame_number}.pose.txt")
                    cv2.imwrite(colour_filename, colour_image)
                    PoseUtil.save_pose(pose_filename, pose)

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
                        subject_from_world: Optional[np.ndarray] = self.__vicon.get_segment_global_pose(
                            subject, subject
                        )

                        # If that succeeds, render the subject based on its type:
                        if subject_from_world is not None:
                            subject_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(subject_from_world)
                            subject_from_source: Optional[np.ndarray] = self.__subject_from_source_cache.get(subject)

                            if subject_from_source is not None:
                                source_from_world: np.ndarray = np.linalg.inv(subject_from_source) @ subject_from_world
                                self.__render_image_source(subject, source_from_world)
                            elif ViconUtil.is_designatable(subject):
                                self.__render_designatable_subject(subject, subject_cam.p(), subject_designations)
                            else:
                                # Treat the subject as a generic one and simply render its Vicon-obtained pose.
                                CameraRenderer.render_camera(subject_cam, axis_scale=0.5)

                # Disable the z-buffer again.
                glPopAttrib()

        # Swap the front and back buffers.
        pygame.display.flip()

    def __render_image_source(self, subject: str, source_from_world: np.ndarray) -> None:
        """
        Render an image source (e.g. a drone).

        :param subject:             The Vicon subject corresponding to the image source.
        :param source_from_world:   A transformation from world space to the space of the image source.
        """
        # Render the coordinate axes of the image source.
        source_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(source_from_world)
        glLineWidth(5)
        CameraRenderer.render_camera(source_cam, axis_scale=0.5)
        glLineWidth(1)

        # If a mesh for the image source is available, render it.
        subject_mesh: Optional[OpenGLTriMesh] = self.__try_get_subject_mesh(subject)
        if subject_mesh is not None:
            world_from_source: np.ndarray = np.linalg.inv(source_from_world)
            with ViconUtil.default_lighting_context():
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(world_from_source)):
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

    # PRIVATE STATIC METHODS

    @staticmethod
    def __load_scene_mesh(scenes_folder: str, scene_timestamp: str, vicon: ViconInterface) -> OpenGLTriMesh:
        """
        Load in a scene mesh, transforming it into the Vicon coordinate system in the process.

        :param scenes_folder:   The folder from which to load the scene mesh.
        :param scene_timestamp: A timestamp indicating which scene mesh to load.
        :param vicon:           The Vicon interface.
        :return:                The scene mesh.
        """
        # Specify the file paths.
        mesh_filename: str = os.path.join(scenes_folder, f"TangoCapture-{scene_timestamp}-cleaned.ply")
        fiducials_filename: str = os.path.join(scenes_folder, f"TangoCapture-{scene_timestamp}-fiducials.txt")

        # Load in the positions of the four ArUco marker corners as estimated during the reconstruction process.
        fiducials: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(fiducials_filename)

        # Stack these positions into a 3x4 matrix.
        p: np.ndarray = np.column_stack([
            fiducials["0_0"],
            fiducials["0_1"],
            fiducials["0_2"],
            fiducials["0_3"]
        ])

        # Look up the Vicon coordinate system positions of the all of the Vicon markers that can currently be seen
        # by the Vicon system, hopefully including ones for the ArUco marker corners.
        marker_positions: Dict[str, np.ndarray] = vicon.get_marker_positions("Registrar")

        # Again, stack the relevant positions into a 3x4 matrix.
        q: np.ndarray = np.column_stack([
            marker_positions["0_0"],
            marker_positions["0_1"],
            marker_positions["0_2"],
            marker_positions["0_3"]
        ])

        # Estimate the rigid transformation between the two sets of points.
        transform: np.ndarray = GeometryUtil.estimate_rigid_transform(p, q)

        # Load in the scene mesh and transform it into the Vicon coordinate system.
        scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_filename)
        scene_mesh_o3d.transform(transform)

        # Convert the scene mesh to OpenGL format and return it.
        return MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d)
