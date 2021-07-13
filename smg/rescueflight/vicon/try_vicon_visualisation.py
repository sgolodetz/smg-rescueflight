import cv2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Callable, Dict, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.meshing import MeshUtil
from smg.opengl import CameraRenderer, OpenGLLightingContext, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.skeletons import Skeleton3D, SkeletonRenderer
from smg.smplx import SMPLBody
from smg.utility import FiducialUtil, GeometryUtil, PooledQueue
from smg.vicon import LiveViconInterface, OfflineViconInterface, SubjectFromSourceCache
from smg.vicon import ViconFrameSaver, ViconInterface, ViconSkeletonDetector


def is_person(subject_name: str, vicon: ViconInterface) -> bool:
    """
    Determine whether or not the specified Vicon subject is a person.

    :param subject_name:    The name of the subject.
    :param vicon:           The Vicon interface.
    :return:                True, if the specified Vicon subject is a person, or False otherwise.
    """
    return "Root" in vicon.get_segment_names(subject_name)


def load_scene_mesh(scenes_folder: str, scene_timestamp: str, vicon: ViconInterface) -> OpenGLTriMesh:
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


def vicon_lighting_context() -> OpenGLLightingContext:
    """
    Get the OpenGL lighting context to use when rendering Vicon scenes.

    :return:    The OpenGL lighting context to use when rendering Vicon scenes.
    """
    direction: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0])
    return OpenGLLightingContext({
        0: OpenGLLightingContext.DirectionalLight(direction),
        1: OpenGLLightingContext.DirectionalLight(-direction),
    })


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--pause", action="store_true",
        help="whether to start the visualisation in its paused state"
    )
    parser.add_argument(
        "--persistence_folder", type=str,
        help="the folder (if any) that should be used for Vicon persistence"
    )
    parser.add_argument(
        "--persistence_mode", type=str, default="none", choices=("input", "none", "output"),
        help="the Vicon persistence mode"
    )
    parser.add_argument(
        "--run_server", action="store_true",
        help="whether to accept connections from mapping clients"
    )
    parser.add_argument(
        "--scenes_folder", type=str, default="C:/spaint/build/bin/apps/spaintgui/meshes",
        help="the folder from which to load the scene mesh"
    )
    parser.add_argument(
        "--scene_timestamp", type=str,
        help="a timestamp indicating which scene mesh to load"
    )
    parser.add_argument(
        "--use_vicon_poses", action="store_true",
        help="whether to use the joint poses produced by the Vicon system"
    )
    args: dict = vars(parser.parse_args())

    persistence_folder: Optional[str] = args["persistence_folder"]
    persistence_mode: str = args["persistence_mode"]

    if persistence_mode != "none" and persistence_folder is None:
        raise RuntimeError(f"Cannot {persistence_mode}: need to specify a persistence folder")
    if persistence_mode == "input" and not os.path.exists(persistence_folder):
        raise RuntimeError("Cannot input: persistence folder does not exist")
    if persistence_mode == "output" and os.path.exists(persistence_folder):
        raise RuntimeError("Cannot output: persistence folder already exists")

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Vicon Visualisation Demo")

    # Set the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the subject-from-source cache.
    subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")

    # TODO: Comment here.
    mapping_server: Optional[MappingServer] = None
    vicon: Optional[ViconInterface] = None

    try:
        # Construct the Vicon interface.
        if persistence_mode == "input":
            vicon = OfflineViconInterface(folder=persistence_folder)
        else:
            vicon = LiveViconInterface()

        # Run a mapping server if requested.
        client_id: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        if args["run_server"]:
            mapping_server = MappingServer(
                frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
                pool_empty_strategy=PooledQueue.PES_REPLACE_RANDOM
            )
            mapping_server.start()

        # If we're in output mode, construct the Vicon frame saver.
        vicon_frame_saver: Optional[ViconFrameSaver] = None
        if persistence_mode == "output":
            vicon_frame_saver = ViconFrameSaver(folder=persistence_folder, vicon=vicon)

        # Construct the skeleton detector.
        skeleton_detector: ViconSkeletonDetector = ViconSkeletonDetector(
            vicon, is_person=is_person, use_vicon_poses=args["use_vicon_poses"]
        )

        # Load the SMPL body models.
        female_body: SMPLBody = SMPLBody(
            "female",
            texture_coords_filename="D:/smplx/textures/smpl/texture_coords.npy",
            texture_image_filename="D:/smplx/textures/smpl/surreal/nongrey_female_0891.jpg"
        )

        male_body: SMPLBody = SMPLBody(
            "male",
            texture_coords_filename="D:/smplx/textures/smpl/texture_coords.npy",
            texture_image_filename="D:/smplx/textures/smpl/surreal/nongrey_male_0170.jpg"
        )

        bodies: Dict[str, SMPLBody] = {
            "Aluna": female_body,
            "Madhu": male_body
        }

        # Initialise the subject mesh cache and specify the subject mesh loaders.
        subject_mesh_cache: Dict[str, OpenGLTriMesh] = {}
        subject_mesh_loaders: Dict[str, Callable[[], OpenGLTriMesh]] = {
            "Tello": lambda: MeshUtil.convert_trimesh_to_opengl(MeshUtil.load_tello_mesh())
        }

        # Load in the scene mesh (if any), transforming it as needed in the process.
        scene_mesh: Optional[OpenGLTriMesh] = None
        scene_timestamp: Optional[str] = args.get("scene_timestamp")
        if scene_timestamp is not None and vicon.get_frame():
            scene_mesh = load_scene_mesh(args["scenes_folder"], scene_timestamp, vicon)

        # Initialise the playback variables.
        pause: bool = args["pause"] or args["run_server"]
        previous_frame_number: Optional[int] = None
        previous_frame_start: Optional[float] = None
        process_next: bool = True

        # Repeatedly:
        while True:
            # Process any PyGame events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # If the user presses the 'b' key, process frames without pausing.
                    if event.key == pygame.K_b:
                        pause = False
                        process_next = True

                    # Otherwise, if the user presses the 'n' key, process the next frame and then pause.
                    elif event.key == pygame.K_n:
                        pause = True
                        process_next = True
                elif event.type == pygame.QUIT:
                    # If the user wants us to quit, shut down pygame.
                    pygame.quit()

                    # Then forcibly terminate the whole process.
                    # noinspection PyProtectedMember
                    os._exit(0)

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

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
                    # Render coordinate axes.
                    CameraRenderer.render_camera(
                        CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
                    )

                    # Render a voxel grid.
                    glColor3f(0.0, 0.0, 0.0)
                    OpenGLUtil.render_voxel_grid([-3, -5, 0], [3, 5, 2], [1, 1, 1], dotted=True)

                    # Render the scene mesh (if any).
                    if scene_mesh is not None:
                        scene_mesh.render()

                    # If we're ready to process the next Vicon frame:
                    if process_next:
                        # Try to get a frame from the Vicon system. If that succeeds:
                        if vicon.get_frame():
                            frame_number: int = vicon.get_frame_number()

                            # If we're running a mapping server, try to get a frame from the client.
                            colour_image: Optional[np.ndarray] = None
                            if mapping_server is not None and mapping_server.has_frames_now(client_id):
                                # # Get the camera parameters from the server.
                                # height, width, _ = mapping_server.get_image_shapes(client_id)[0]
                                # image_size = (width, height)
                                # intrinsics = mapping_server.get_intrinsics(client_id)[0]

                                # Get the newest frame from the mapping server.
                                mapping_server.peek_newest_frame(client_id, receiver)
                                colour_image = receiver.get_rgb_image()

                                # Show the image.
                                cv2.imshow("Received Image", colour_image)
                                cv2.waitKey(1)

                            # If we're in output mode:
                            if True:  # persistence_mode == "output":
                                # If we aren't running a mapping server:
                                if mapping_server is None:
                                    # Save the Vicon frame to disk.
                                    # vicon_frame_saver.save_frame()
                                    print("Would save Vicon frame")

                                # Otherwise, if we are running a server and an image has been obtained from the client:
                                elif colour_image is not None:
                                    # Save the Vicon frame to disk.
                                    # vicon_frame_saver.save_frame()
                                    print("Would save Vicon frame")

                                    # Save the colour image to disk.
                                    filename: str = os.path.join(persistence_folder, f"{frame_number}.png")
                                    print(f"Would save image to {filename}")

                            # Check how long has elapsed since the start of the previous frame. If it's not long
                            # enough, pause until the expected amount of time has elapsed.
                            frame_start: float = timer()
                            if previous_frame_number is not None:
                                recording_fps: int = 200
                                expected_time_delta: float = (frame_number - previous_frame_number) / recording_fps
                                time_delta: float = frame_start - previous_frame_start
                                time_delta_offset: float = expected_time_delta - time_delta
                                if time_delta_offset > 0:
                                    time.sleep(time_delta_offset)

                            previous_frame_number = frame_number
                            previous_frame_start = frame_start

                    # Print out the frame number.
                    print(f"=== Frame {vicon.get_frame_number()} ===")

                    # For each Vicon subject:
                    for subject in vicon.get_subject_names():
                        # Render all of its markers.
                        for marker_name, marker_pos in vicon.get_marker_positions(subject).items():
                            glColor3f(1.0, 0.0, 0.0)
                            OpenGLUtil.render_sphere(marker_pos, 0.014, slices=10, stacks=10)

                        # If the subject is a person, don't bother trying to render its (rigid-body) pose.
                        if is_person(subject, vicon):
                            continue

                        # Otherwise, assume it's a single-segment subject and try to get its pose.
                        subject_from_world: Optional[np.ndarray] = vicon.get_segment_global_pose(subject, subject)

                        # If that succeeds:
                        if subject_from_world is not None:
                            # Assume that the subject corresponds to an image source, and try to get the
                            # relative transformation from that image source to the subject.
                            subject_from_source: Optional[np.ndarray] = subject_from_source_cache.get(subject)

                            # If that succeeds (i.e. the subject does correspond to an image source, and we know the
                            # relative transformation):
                            if subject_from_source is not None:
                                # Render the pose of the image source.
                                source_from_world: np.ndarray = np.linalg.inv(subject_from_source) @ subject_from_world
                                source_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(source_from_world)
                                glLineWidth(5)
                                CameraRenderer.render_camera(source_cam, axis_scale=0.5)
                                glLineWidth(1)

                                # Try to look up the mesh for the subject in the cache.
                                subject_mesh: Optional[OpenGLTriMesh] = subject_mesh_cache.get(subject)

                                # If it's not there, try to load it into the cache.
                                if subject_mesh is None:
                                    subject_mesh_loader: Optional[Callable[[], OpenGLTriMesh]] = \
                                        subject_mesh_loaders.get(subject)
                                    if subject_mesh_loader is not None:
                                        subject_mesh = subject_mesh_loader()
                                        subject_mesh_cache[subject] = subject_mesh

                                # If the mesh for the subject is now available (one way or the other), render it.
                                if subject_mesh is not None:
                                    world_from_source: np.ndarray = np.linalg.inv(source_from_world)
                                    with vicon_lighting_context():
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
                    skeletons: Dict[str, Skeleton3D] = skeleton_detector.detect_skeletons()

                    # Render the skeletons and their corresponding SMPL bodies.
                    for subject, skeleton in skeletons.items():
                        with vicon_lighting_context():
                            SkeletonRenderer.render_skeleton(skeleton)

                            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                            body: SMPLBody = bodies.get(subject, male_body)
                            body.render_from_skeleton(skeleton)
                            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                        SkeletonRenderer.render_keypoint_poses(skeleton)

                    # Decide whether to continue processing subsequent frames or wait.
                    process_next = not pause

            # Swap the front and back buffers.
            pygame.display.flip()
    finally:
        # Terminate the mapping server.
        if mapping_server is not None:
            mapping_server.terminate()

        # Terminate the Vicon system.
        if vicon is not None:
            vicon.terminate()


if __name__ == "__main__":
    main()
