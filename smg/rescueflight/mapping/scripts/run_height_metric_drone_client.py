import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil
from smg.comms.mapping import MappingClient
from smg.joysticks import FutabaT6K
from smg.mapping.metric import HeightMetricDroneFSM
from smg.opengl import CameraRenderer, OpenGLImageRenderer, OpenGLMatrixContext, OpenGLUtil
from smg.pyorbslam2 import MonocularTracker
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.rotory import DroneFactory
from smg.utility import ImageUtil, TrajectorySmoother


def render_window(*, drone_image: np.ndarray, image_renderer: OpenGLImageRenderer,
                  trajectory: List[Tuple[float, np.ndarray]],
                  viewing_pose: np.ndarray, window_size: Tuple[int, int]) -> None:
    """
    Render the application window.

    :param drone_image:     The most recent image from the drone.
    :param image_renderer:  An OpenGL-based image renderer.
    :param trajectory:      The metric trajectory of the drone, as estimated by the tracker.
    :param viewing_pose:    The pose from which the scene is being viewed.
    :param window_size:     The application window size, as a (width, height) tuple.
    """
    # Clear the window.
    OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), window_size)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render the drone image.
    OpenGLUtil.set_viewport((0.0, 0.0), (0.5, 1.0), window_size)
    image_renderer.render_image(ImageUtil.flip_channels(drone_image))

    # Render the drone's trajectories in 3D.
    OpenGLUtil.set_viewport((0.5, 0.0), (1.0, 1.0), window_size)

    glDepthFunc(GL_LEQUAL)
    glEnable(GL_DEPTH_TEST)

    with OpenGLMatrixContext(
        GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix((500.0, 500.0, 320.0, 240.0), 640, 480)
    ):
        with OpenGLMatrixContext(
            GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(CameraPoseConverter.pose_to_modelview(viewing_pose))
        ):
            glColor3f(0.0, 0.0, 0.0)
            OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

            origin: SimpleCamera = SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0])
            CameraRenderer.render_camera(origin, body_colour=(1.0, 1.0, 0.0), body_scale=0.1)

            OpenGLUtil.render_trajectory(trajectory, colour=(0.0, 0.0, 1.0))

    glDisable(GL_DEPTH_TEST)

    # Swap the buffers.
    pygame.display.flip()


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        help="an optional directory into which to save output files"
    )
    parser.add_argument(
        "--reconstruct", "-r", action="store_true",
        help="whether to connect to the mapping server to reconstruct a map"
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="whether to save the sequence of frames that have been obtained from the drone"
    )
    args: dict = vars(parser.parse_args())

    # Initialise pygame and its joystick module.
    pygame.init()
    pygame.joystick.init()

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

    # Construct the drone.
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    drone_type: str = args.get("drone_type")

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        # Create the window.
        window_size: Tuple[int, int] = (1280, 480)
        pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)

        # Construct the image renderer.
        with OpenGLImageRenderer() as image_renderer:
            # Construct the tracker.
            with MonocularTracker(
                settings_file=f"settings-{drone_type}.yaml", use_viewer=True,
                voc_file="C:/orbslam2/Vocabulary/ORBvoc.bin", wait_till_ready=False
            ) as tracker:
                # Construct and calibrate the Futaba T6K.
                joystick: FutabaT6K = FutabaT6K(joystick_idx)
                joystick.calibrate()

                # Construct the camera controller.
                camera_controller: KeyboardCameraController = KeyboardCameraController(
                    SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05,
                    canonical_linear_speed=0.1
                )

                # Construct the state machine for the drone.
                mapping_client: Optional[MappingClient] = None
                if args["reconstruct"]:
                    mapping_client = MappingClient(frame_compressor=RGBDFrameMessageUtil.compress_frame_message)

                state_machine: HeightMetricDroneFSM = HeightMetricDroneFSM(
                    drone, joystick, mapping_client,
                    output_dir=args.get("output_dir"),
                    save_frames=args.get("save_frames")
                )

                # Initialise the timestamp and the drone's trajectory smoother (used for visualisation).
                timestamp: float = 0.0
                trajectory_smoother: TrajectorySmoother = TrajectorySmoother()

                # While the state machine is still running:
                while state_machine.alive():
                    # Process any pygame events.
                    takeoff_requested: bool = False
                    landing_requested: bool = False

                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN:
                            if event.button == 0:
                                takeoff_requested = True
                        elif event.type == pygame.JOYBUTTONUP:
                            if event.button == 0:
                                landing_requested = True
                        elif event.type == pygame.QUIT:
                            state_machine.terminate()

                    # If the user closed the application and the state machine terminated, early out.
                    if not state_machine.alive():
                        break

                    # Allow the user to control the camera.
                    camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                    # Get an image from the drone.
                    image: np.ndarray = drone.get_image()

                    # Try to estimate a transformation from initial camera space to current camera space
                    # using the tracker.
                    tracker_c_t_i: Optional[np.ndarray] = tracker.estimate_pose(image) if tracker.is_ready() else None

                    # Run an iteration of the state machine.
                    state_machine.iterate(
                        image, drone.get_intrinsics(), tracker_c_t_i, drone.get_height(),
                        takeoff_requested, landing_requested
                    )

                    # Update the drone's trajectory.
                    tracker_w_t_c: Optional[np.ndarray] = state_machine.get_tracker_w_t_c()
                    if tracker_w_t_c is not None:
                        trajectory_smoother.append(timestamp, tracker_w_t_c)

                    # Update the caption of the window to reflect the current state.
                    pygame.display.set_caption(
                        "Height-Based Metric Drone Client: "
                        f"State = {int(state_machine.get_state())}; "
                        f"Battery Level = {drone.get_battery_level()}; "
                        f"Height = {drone.get_height()}"
                    )

                    # Render the contents of the window.
                    render_window(
                        drone_image=image,
                        image_renderer=image_renderer,
                        trajectory=trajectory_smoother.get_smoothed_trajectory()[::10],
                        viewing_pose=camera_controller.get_pose(),
                        window_size=window_size
                    )

                    # Update the timestamp.
                    timestamp += 1.0

                # If the tracker's not ready yet, forcibly terminate the whole process (this isn't graceful, but
                # if we don't do it then we may have to wait a very long time for it to finish initialising).
                if not tracker.is_ready():
                    # noinspection PyProtectedMember
                    os._exit(0)

    # Shut down pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
