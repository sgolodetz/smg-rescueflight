import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

from smg.comms.mapping import MappingServer, RGBDFrameReceiver, RGBDFrameMessageUtil
from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.pyopenpose import BoneLengthEstimator, SkeletonDetector
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Skeleton, SkeletonRenderer
from smg.utility import GeometryUtil, PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("RGB-D Skeleton Detection Server")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.PES_REPLACE_RANDOM
    ) as server:
        client_id: int = 0
        image_size: Optional[Tuple[int, int]] = None
        intrinsics: Optional[Tuple[float, float, float, float]] = None
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        skeletons_3d: List[Skeleton] = []

        # Construct the skeleton detector.
        params: Dict[str, Any] = {"model_folder": "D:/openpose-1.6.0/models/"}
        with SkeletonDetector(params) as skeleton_detector:
            # Construct the bone length estimator.
            bone_length_estimator: BoneLengthEstimator = BoneLengthEstimator()

            # Start the server.
            server.start()

            while True:
                # Process any PyGame events.
                for event in pygame.event.get():
                    # If the user wants us to quit:
                    if event.type == pygame.QUIT:
                        # Shut down pygame, and destroy any OpenCV windows.
                        pygame.quit()
                        cv2.destroyAllWindows()

                        # Forcibly terminate the whole process.
                        # noinspection PyProtectedMember
                        os._exit(0)

                # If the server has a frame from the client that has not yet been processed:
                if server.has_frames_now(client_id):
                    # Get the camera parameters from the server.
                    height, width, _ = server.get_image_shapes(client_id)[0]
                    image_size = (width, height)
                    intrinsics = server.get_intrinsics(client_id)[0]

                    # Get the newest frame from the server.
                    server.peek_newest_frame(client_id, receiver)
                    colour_image: np.ndarray = receiver.get_rgb_image()
                    depth_image: np.ndarray = receiver.get_depth_image()
                    tracker_w_t_c: np.ndarray = receiver.get_pose()

                    # Use OpenPose to detect 2D skeletons in the colour image.
                    skeletons_2d, output_image = skeleton_detector.detect_skeletons_2d(colour_image)

                    # Compute the world-space points image and the mask indicating which pixels are valid.
                    ws_points: np.ndarray = GeometryUtil.compute_world_points_image_fast(
                        depth_image, tracker_w_t_c, intrinsics
                    )

                    mask: np.ndarray = np.where(depth_image != 0, 255, 0).astype(np.uint8)

                    # Lift the 2D skeletons into 3D.
                    skeletons_3d = skeleton_detector.lift_skeletons_to_3d(skeletons_2d, ws_points, mask)

                    # If there's only one skeleton in the frame:
                    # TODO: Once we can associate skeletons between frames, this limitation can be relaxed.
                    if len(skeletons_3d) == 1:
                        # Use the skeleton to update the expected bone lengths.
                        bone_length_estimator.add_estimates(skeletons_3d[0])

                        # Get the expected bone lengths, and print them out.
                        expected_bone_lengths: Dict[Tuple[str, str], float] = \
                            bone_length_estimator.get_expected_bone_lengths()

                        print(expected_bone_lengths)

                        # Remove any bones that have unexpected lengths from the skeleton.
                        skeletons_3d[0] = SkeletonDetector.remove_bad_bones(skeletons_3d[0], expected_bone_lengths)

                    # Blend the OpenPose output image with the depth image.
                    depth_image_uc: np.ndarray = np.clip(depth_image * 255 / 5, 0, 255).astype(np.uint8)
                    blended_image: np.ndarray = np.zeros(colour_image.shape, dtype=np.uint8)
                    for i in range(3):
                        blended_image[:, :, i] = (output_image[:, :, i] * 0.5 + depth_image_uc * 0.5).astype(np.uint8)

                    # Show the result to aid debugging.
                    cv2.imshow("2D OpenPose Result", blended_image)
                    cv2.waitKey(1)

                # Allow the user to control the camera.
                camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                # Clear the colour and depth buffers.
                glClearColor(1.0, 1.0, 1.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Once at least one frame has been received:
                if image_size is not None:
                    # Set the projection matrix.
                    with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                        GeometryUtil.rescale_intrinsics(intrinsics, image_size, window_size), *window_size
                    )):
                        # Set the model-view matrix.
                        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                            CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
                        )):
                            # Render a voxel grid.
                            glColor3f(0.0, 0.0, 0.0)
                            OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                            # Render the 3D skeletons.
                            for skeleton_3d in skeletons_3d:
                                SkeletonRenderer.render_skeleton(skeleton_3d)

                # Swap the front and back buffers.
                pygame.display.flip()


if __name__ == "__main__":
    main()
