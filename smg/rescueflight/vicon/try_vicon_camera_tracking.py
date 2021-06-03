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
from smg.utility import ImageUtil, PoseUtil
from smg.vicon import ViconInterface


def render_window(*, image: np.ndarray, image_renderer: OpenGLImageRenderer, source_from_world: Optional[np.ndarray],
                  vicon: ViconInterface, viewing_pose: np.ndarray, window_size: Tuple[int, int]) \
        -> None:
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
            glLineWidth(5)
            CameraRenderer.render_camera(
                CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
            )
            glLineWidth(1)

            # Render a voxel grid.
            glColor3f(0.0, 0.0, 0.0)
            OpenGLUtil.render_voxel_grid([-3, -5, 0], [3, 5, 2], [1, 1, 1], dotted=True)

            # Render coordinate axes for the image source (if its current pose is known).
            if source_from_world is not None:
                glLineWidth(5)
                CameraRenderer.render_camera(
                    CameraPoseConverter.pose_to_camera(source_from_world),
                    # axes_type=CameraRenderer.AXES_NUV
                )
                glLineWidth(1)

            # For each Vicon subject:
            for subject in vicon.get_subject_names():
                # Render all of its markers.
                for marker_name, marker_pos in vicon.get_marker_positions(subject).items():
                    print(marker_name, marker_pos)
                    glColor3f(1.0, 0.0, 0.0)
                    OpenGLUtil.render_sphere(marker_pos, 0.014, slices=10, stacks=10)

                # Assume it's a single-segment subject and try to render its coordinate axes.
                subject_pose: Optional[np.ndarray] = vicon.get_segment_pose(subject, subject)
                if subject_pose is not None:
                    subject_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(subject_pose)
                    CameraRenderer.render_camera(subject_cam, axis_scale=0.5)

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

        # TODO
        source_from_world: Optional[np.ndarray] = None

        # TODO
        subject_from_source: np.ndarray = PoseUtil.load_pose("Tello.txt")

        # Connect to the Vicon system.
        with ViconInterface() as vicon:
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

                    # Try to get a frame of Vicon data. If it's available:
                    if vicon.get_frame():
                        # Print out the frame number.
                        print(f"=== Frame {vicon.get_frame_number()} ===")

                        # TODO
                        subject_from_world: Optional[np.ndarray] = vicon.get_segment_pose("Tello", "Tello")
                        if subject_from_world is not None:
                            source_from_world = np.linalg.inv(subject_from_source) @ subject_from_world
                    else:
                        source_from_world = None

                    # Render the contents of the window.
                    render_window(
                        image=image,
                        image_renderer=image_renderer,
                        source_from_world=source_from_world,
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
