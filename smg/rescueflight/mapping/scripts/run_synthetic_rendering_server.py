import cv2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.meshing import MeshUtil
from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.utility import GeometryUtil, PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Synthetic Rendering Server")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Enable back-face culling.
    glFrontFace(GL_CCW)
    glEnable(GL_CULL_FACE)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Load in the scene mesh.
    scene_mesh: Optional[OpenGLTriMesh] = None
    mesh_filename: str = "D:/Apple/2022_03_20_10_54_36-smg/untextured_output.ply"
    if os.path.exists(mesh_filename):
        # noinspection PyUnresolvedReferences
        scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_filename)
        scene_mesh = MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d)

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.PES_REPLACE_RANDOM
    ) as server:
        client_id: int = 0
        image_size: Optional[Tuple[int, int]] = None
        intrinsics: Optional[Tuple[float, float, float, float]] = None
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        tracker_w_t_c: Optional[np.ndarray] = None

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
                # colour_image: np.ndarray = receiver.get_rgb_image()
                tracker_w_t_c = receiver.get_pose()

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
                        # CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
                        # tracker_w_t_c
                        CameraPoseConverter.pose_to_modelview(np.linalg.inv(tracker_w_t_c))
                    )):
                        # Render a voxel grid.
                        glColor3f(0.0, 0.0, 0.0)
                        OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                        # Render coordinate axes.
                        CameraRenderer.render_camera(CameraUtil.make_default_camera())

                        # Render the scene mesh.
                        if scene_mesh is not None:
                            scene_mesh.render()

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
