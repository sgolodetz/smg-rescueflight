import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, Optional, Tuple

from smg.imagesources import RGBFromRGBDImageSource, RGBImageSource
from smg.opengl import CameraRenderer, OpenGLImageRenderer, OpenGLMatrixContext, OpenGLUtil
from smg.openni import OpenNICamera, OpenNIRGBDImageSource
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotory import DroneFactory, DroneRGBImageSource
from smg.utility import ImageUtil
from smg.vicon import LiveViconInterface, SubjectFromSourceCache, ViconInterface


def render_window(*, image: np.ndarray, image_renderer: OpenGLImageRenderer,
                  subject_from_source_cache: SubjectFromSourceCache, vicon: ViconInterface,
                  viewing_pose: np.ndarray, window_size: Tuple[int, int]) -> None:
    """
    Render the application window.

    :param image:                       The most recent image from the camera.
    :param image_renderer:              An OpenGL-based image renderer.
    :param subject_from_source_cache:   A cache of the transformations from image sources to their Vicon subjects.
    :param vicon:                       The Vicon interface.
    :param viewing_pose:                The pose from which the scene is being viewed.
    :param window_size:                 The application window size, as a (width, height) tuple.
    """
    # Clear the window.
    OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), window_size)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render the Vicon scene.
    OpenGLUtil.set_viewport((0.0, 0.0), (0.5, 1.0), window_size)

    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)
    width, height = window_size
    with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
        intrinsics, width // 2, height
    )):
        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
            CameraPoseConverter.pose_to_modelview(viewing_pose)
        )):
            # Render coordinate axes.
            CameraRenderer.render_camera(
                CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
            )

            # Render a voxel grid.
            glColor3f(0.0, 0.0, 0.0)
            OpenGLUtil.render_voxel_grid([-3, -5, 0], [3, 5, 2], [1, 1, 1], dotted=True)

            # If a frame of Vicon data is available:
            if vicon.get_frame():
                # Print out the frame number.
                print(f"=== Frame {vicon.get_frame_number()} ===")

                # For each Vicon subject:
                for subject in vicon.get_subject_names():
                    # Render all of its markers.
                    for marker_name, marker_pos in vicon.get_marker_positions(subject).items():
                        print(marker_name, marker_pos)
                        glColor3f(1.0, 0.0, 0.0)
                        OpenGLUtil.render_sphere(marker_pos, 0.014, slices=10, stacks=10)

                    # Assume it's a single-segment subject and try to get its pose from the Vicon system.
                    subject_from_world: Optional[np.ndarray] = vicon.get_segment_global_pose(subject, subject)

                    # If that succeeds:
                    if subject_from_world is not None:
                        # Render the subject pose obtained from the Vicon system.
                        subject_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(subject_from_world)
                        CameraRenderer.render_camera(subject_cam, axis_scale=0.5)

                        # Assume that the subject corresponds to an image source, and try to get the
                        # relative transformation from that image source to the subject.
                        subject_from_source: Optional[np.ndarray] = subject_from_source_cache.get(subject)

                        # If that succeeds (i.e. it does correspond to an image source, and we know the
                        # relative transformation):
                        if subject_from_source is not None:
                            # Render the pose of the image source as well.
                            source_from_world: np.ndarray = \
                                np.linalg.inv(subject_from_source) @ subject_from_world
                            source_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(source_from_world)
                            glLineWidth(5)
                            CameraRenderer.render_camera(source_cam, axis_scale=0.5)
                            glLineWidth(1)

    # Render the image.
    OpenGLUtil.set_viewport((0.5, 0.0), (1.0, 1.0), window_size)
    image_renderer.render_image(ImageUtil.flip_channels(image))

    # Swap the front and back buffers.
    pygame.display.flip()


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--source_type", "-t", type=str, required=True, choices=("ardrone2", "kinect", "tello"),
        help="the input type"
    )
    args: dict = vars(parser.parse_args())

    image_source: Optional[RGBImageSource] = None
    try:
        # Construct the RGB image source.
        # FIXME: This is duplicate code - factor it out.
        source_type: str = args["source_type"]
        if source_type == "kinect":
            image_source = RGBFromRGBDImageSource(OpenNIRGBDImageSource(OpenNICamera(mirror_images=True)))
        else:
            kwargs: Dict[str, dict] = {
                "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
                "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
            }
            image_source = DroneRGBImageSource(DroneFactory.make_drone(source_type, **kwargs[source_type]))

        # Initialise PyGame and create the window.
        pygame.init()
        window_size: Tuple[int, int] = (1280, 480)
        pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Vicon Camera Tracking Demo")

        # Enable the z-buffer.
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Construct the camera controller.
        camera_controller: KeyboardCameraController = KeyboardCameraController(
            SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
        )

        # Construct the subject-from-source cache.
        subject_from_source_cache: SubjectFromSourceCache = SubjectFromSourceCache(".")

        # Connect to the Vicon system.
        with LiveViconInterface() as vicon:
            # Construct the image renderer.
            with OpenGLImageRenderer() as image_renderer:
                # Repeatedly:
                while True:
                    # Process any PyGame events.
                    for event in pygame.event.get():
                        # If the user wants us to quit:
                        if event.type == pygame.QUIT:
                            # Shut down pygame.
                            pygame.quit()

                            # Forcibly terminate the whole process.
                            # noinspection PyProtectedMember
                            os._exit(0)

                    # Allow the user to control the camera.
                    camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                    # Get an image from the camera.
                    image: np.ndarray = image_source.get_image()

                    # Render the contents of the window.
                    render_window(
                        image=image,
                        image_renderer=image_renderer,
                        subject_from_source_cache=subject_from_source_cache,
                        vicon=vicon,
                        viewing_pose=camera_controller.get_pose(),
                        window_size=window_size
                    )
    finally:
        # Terminate the image source.
        if image_source is not None:
            image_source.terminate()


if __name__ == "__main__":
    main()
