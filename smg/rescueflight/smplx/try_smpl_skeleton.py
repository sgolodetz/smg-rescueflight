import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

from smg.comms.skeletons import RemoteSkeletonDetector
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLUtil
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.skeletons import Skeleton3D, SkeletonRenderer
from smg.smplx import SMPLBody
from smg.utility import CameraParameters, GeometryUtil, RGBDSequenceUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the directory from which to load the sequence"
    )
    args: dict = vars(parser.parse_args())

    sequence_dir: str = args["sequence_dir"]

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("SMPL Skeleton Demo")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the SMPL body.
    body: SMPLBody = SMPLBody(
        "female",
        texture_coords_filename="D:/smplx/textures/smpl/texture_coords.npy",
        texture_image_filename="D:/smplx/textures/smpl/surreal/nongrey_female_0891.jpg"
    )

    try:
        # Construct the remote skeleton detector.
        with RemoteSkeletonDetector() as skeleton_detector:
            # Try to load the camera parameters for the sequence. If this fails, raise an exception.
            calib: Optional[CameraParameters] = RGBDSequenceUtil.try_load_calibration(sequence_dir)
            if calib is None:
                raise RuntimeError(f"Cannot load calibration from '{sequence_dir}'")

            intrinsics: Tuple[float, float, float, float] = calib.get_intrinsics("colour")
            image_size: Tuple[int, int] = calib.get_image_size("colour")

            # Initialise a few variables.
            colour_image: Optional[np.ndarray] = None
            frame_idx: int = 0
            pause: bool = True
            process_next: bool = True
            skeletons: List[Skeleton3D] = []

            # Repeatedly:
            while True:
                # Process any PyGame events.
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        # If the user presses the 'b' key, process the sequence without pausing.
                        if event.key == pygame.K_b:
                            pause = False
                            process_next = True

                        # Otherwise, if the user presses the 'n' key, process the next image and then pause.
                        elif event.key == pygame.K_n:
                            pause = True
                            process_next = True
                    elif event.type == pygame.QUIT:
                        # If the user wants us to quit, shut down pygame, and destroy any OpenCV windows.
                        pygame.quit()

                        # Then forcibly terminate the whole process.
                        # noinspection PyProtectedMember
                        os._exit(0)

                # Try to load an RGB-D frame from disk.
                frame: Optional[Dict[str, Any]] = RGBDSequenceUtil.try_load_frame(frame_idx, sequence_dir)

                # If the frame was successfully loaded and we're processing it now:
                if frame is not None and process_next:
                    colour_image = frame["colour_image"]

                    # Detect 3D skeletons in the colour image.
                    start = timer()
                    skeletons, _ = skeleton_detector.detect_skeletons(colour_image, frame["world_from_camera"])
                    end = timer()
                    print(f"Skeleton Detection Time: {end - start}s")

                    # Advance to the next frame.
                    frame_idx += 1

                    # Decide whether to continue processing subsequent frames or wait.
                    process_next = not pause

                # Show the most recent colour image (if any) so that the user can see what's going on.
                if colour_image is not None:
                    cv2.imshow("Colour Image", colour_image)
                    cv2.waitKey(1)

                # Allow the user to control the camera.
                camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                # Clear the colour and depth buffers.
                glClearColor(1.0, 1.0, 1.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Set the projection matrix.
                with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                    GeometryUtil.rescale_intrinsics(intrinsics, image_size, window_size), *window_size
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
                        OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                        # Render the people.
                        with SkeletonRenderer.default_lighting_context():
                            for skeleton in skeletons:
                                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

                                SkeletonRenderer.render_skeleton(skeleton)
                                SkeletonRenderer.render_keypoint_orienters(skeleton)
                                SkeletonRenderer.render_keypoint_poses(skeleton)
                                body.render_from_skeleton(skeleton)

                                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                                body.render_joints()

                # Swap the front and back buffers.
                pygame.display.flip()
    except RuntimeError as e:
        # If any exception is raised, print it out so that we can see what happened.
        print(e)


if __name__ == "__main__":
    main()
