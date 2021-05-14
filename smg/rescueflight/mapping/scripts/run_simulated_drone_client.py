import cv2
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import sys

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Optional, Tuple

from smg.opengl import CameraRenderer, OpenGLFrameBuffer, OpenGLMatrixContext, OpenGLUtil, TriangleMesh
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotory.drones import SimulatedDrone


class MeshRenderer:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, mesh: TriangleMesh):
        self.__framebuffer: Optional[OpenGLFrameBuffer] = None
        self.__mesh: TriangleMesh = mesh

    # PUBLIC METHODS

    def render(self) -> None:
        # Enable lighting.
        glEnable(GL_LIGHTING)

        # Set up the first directional light.
        glEnable(GL_LIGHT0)
        pos = np.array([0.0, -2.0, -1.0, 0.0])  # type: np.ndarray
        glLightfv(GL_LIGHT0, GL_POSITION, pos)

        # Set up the second directional light.
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, np.array([1, 1, 1, 1]))
        glLightfv(GL_LIGHT1, GL_SPECULAR, np.array([1, 1, 1, 1]))
        glLightfv(GL_LIGHT1, GL_POSITION, -pos)

        # Enable colour-based materials (i.e. let material properties be defined by glColor).
        glEnable(GL_COLOR_MATERIAL)

        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)

        # TODO
        self.__mesh.render()

        glDisable(GL_CULL_FACE)

        # Disable colour-based materials and lighting again.
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHTING)

    def render_to_image(self, world_from_camera: np.ndarray, image_size: Tuple[int, int],
                        intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        # If the OpenGL framebuffer hasn't been constructed yet, construct it now.
        # FIXME: Support image size changes.
        width, height = image_size
        if self.__framebuffer is None:
            self.__framebuffer = OpenGLFrameBuffer(width, height)

        # TODO
        with self.__framebuffer:
            # Set the viewport to encompass the whole framebuffer.
            OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), (width, height))

            # Clear the background to black.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set the projection matrix.
            with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                intrinsics, width, height
            )):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(np.linalg.inv(world_from_camera))
                )):
                    self.render()
                    return OpenGLUtil.read_bgr_image(width, height)


def main() -> None:
    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Simulated Drone Client")

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Load the mesh.
    o3d_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(
        "C:/spaint/build/bin/apps/spaintgui/meshes/groundtruth-decimated.ply"
    )
    o3d_mesh.compute_vertex_normals(True)
    mesh: TriangleMesh = TriangleMesh(
        np.asarray(o3d_mesh.vertices),
        np.asarray(o3d_mesh.vertex_colors),
        np.asarray(o3d_mesh.triangles),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals)
    )

    # TODO
    mesh_renderer: MeshRenderer = MeshRenderer(mesh)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.025
    )

    with SimulatedDrone(
            image_renderer=mesh_renderer.render_to_image,
            image_size=(640, 480),
            intrinsics=intrinsics
    ) as drone:
        # Repeatedly:
        while True:
            # Process any PyGame events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)

            # TODO
            drone.set_pose(np.linalg.inv(camera_controller.get_pose()))
            image, world_from_camera = drone.get_image_and_pose()
            print(world_from_camera)
            cv2.imshow("Image", image)
            cv2.waitKey(1)

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Determine the viewing pose.
            viewing_pose: np.ndarray = camera_controller.get_pose()

            # Set the projection matrix.
            with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(intrinsics, *window_size)):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(viewing_pose)
                )):
                    # Render a voxel grid.
                    glColor3f(0.0, 0.0, 0.0)
                    OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                    # Render coordinate axes.
                    CameraRenderer.render_camera(
                        CameraUtil.make_default_camera(), body_colour=(1.0, 1.0, 0.0), body_scale=0.1
                    )

                    # TODO
                    mesh_renderer.render()

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
