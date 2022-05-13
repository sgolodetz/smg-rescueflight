import cv2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from typing import Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.meshing import MeshUtil
from smg.opengl import OpenGLDepthTestingContext, OpenGLFrameBuffer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import CameraParameters, PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--calibration_filename", type=str, required=True,
        help="the name of the file containing the camera calibration parameters"
    )
    parser.add_argument(
        "--max_depth", type=float, default=4.0,
        help="the maximum depth values (in m) to keep"
    )
    parser.add_argument(
        "--scene_mesh_filename", type=str, required=True,
        help="the name of the file containing the scene mesh"
    )
    args: dict = vars(parser.parse_args())

    calibration_filename: str = args["calibration_filename"]
    max_depth: float = args["max_depth"]
    scene_mesh_filename: str = args["scene_mesh_filename"]

    # Try to load in the camera calibration parameters.
    calib: Optional[CameraParameters] = CameraParameters.try_load(calibration_filename)
    if calib is None:
        raise RuntimeError(f"Error: Could not load camera parameters from {calibration_filename}")

    image_size: Tuple[int, int] = calib.get_image_size("colour")
    intrinsics: Optional[Tuple[float, float, float, float]] = calib.get_intrinsics("colour")

    # Initialise PyGame and create a hidden window so that we can use OpenGL.
    pygame.init()
    window_size: Tuple[int, int] = (1, 1)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.HIDDEN | pygame.OPENGL)

    # Set up an off-screen frame-buffer.
    framebuffer: OpenGLFrameBuffer = OpenGLFrameBuffer(*image_size)
    with framebuffer:
        # Set the viewport to encompass the whole frame-buffer.
        OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), image_size)

        # Enable back-face culling.
        glFrontFace(GL_CCW)
        glEnable(GL_CULL_FACE)

    # Try to load in the scene mesh.
    # noinspection PyUnusedLocal
    scene_mesh: Optional[OpenGLTriMesh] = None
    if os.path.exists(scene_mesh_filename):
        # noinspection PyUnresolvedReferences
        scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(scene_mesh_filename)
        scene_mesh = MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d)

    if scene_mesh is None:
        raise RuntimeError(f"Error: Could not load scene mesh from {scene_mesh_filename}")

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.PES_WAIT
    ) as server:
        client_id: int = 0
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        tracker_w_t_c: Optional[np.ndarray] = None

        # Start the server.
        server.start()

        while True:
            # If the server has a frame from the client that has not yet been processed:
            if server.has_frames_now(client_id):
                # Get the oldest frame from the server and extract its pose.
                server.get_frame(client_id, receiver)
                tracker_w_t_c = receiver.get_pose()

            # Once at least one frame has been received:
            if tracker_w_t_c is not None:
                # Enable the frame-buffer.
                with framebuffer:
                    # Enable depth testing.
                    with OpenGLDepthTestingContext(GL_LEQUAL):
                        # Clear the colour and depth buffers.
                        glClearColor(0.0, 0.0, 0.0, 1.0)
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                        # Set the projection matrix.
                        with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                            intrinsics, *image_size
                        )):
                            # Set the model-view matrix.
                            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                                CameraPoseConverter.pose_to_modelview(np.linalg.inv(tracker_w_t_c))
                            )):
                                # Render the scene mesh.
                                scene_mesh.render()

                        # Read the synthetic colour and depth images from the frame-buffer.
                        colour_image: np.ndarray = OpenGLUtil.read_bgr_image(*image_size)
                        depth_image: np.ndarray = OpenGLUtil.read_depth_image(*image_size)
                        depth_image[depth_image > max_depth] = 0.0

                # Show the synthetic colour and depth images.
                cv2.imshow("Colour Image", colour_image)
                cv2.imshow("Depth Image", depth_image / 5)
                c: int = cv2.waitKey(1)

                # If the user presses 'q':
                if c == ord('q'):
                    # Shut down pygame, and destroy any OpenCV windows.
                    pygame.quit()
                    cv2.destroyAllWindows()

                    # Forcibly terminate the whole process.
                    # noinspection PyProtectedMember
                    os._exit(0)


if __name__ == "__main__":
    main()
