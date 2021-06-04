import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, Optional, Tuple

from smg.opengl import CameraRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.utility import FiducialUtil, GeometryUtil
from smg.vicon import SubjectFromSourceCache, ViconInterface


# FIXME: This is a temporary copy of the one from DroneSimulator - it should be put somewhere more central.
def convert_trimesh_to_opengl(o3d_mesh: o3d.geometry.TriangleMesh) -> OpenGLTriMesh:
    """
    Convert an Open3D triangle mesh to an OpenGL one.

    :param o3d_mesh:    The Open3D triangle mesh.
    :return:            The OpenGL mesh.
    """
    # FIXME: This should probably be moved somewhere more central at some point.
    o3d_mesh.compute_vertex_normals(True)
    return OpenGLTriMesh(
        np.asarray(o3d_mesh.vertices),
        np.asarray(o3d_mesh.vertex_colors),
        np.asarray(o3d_mesh.triangles),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals)
    )


def estimate_scene_transformation(vicon: ViconInterface) -> np.ndarray:
    # Load in the positions of the four marker corners as estimated during the ground-truth reconstruction.
    fiducials: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(
        "C:/spaint/build/bin/apps/spaintgui/meshes/TangoCapture-20210604-124834-fiducials.txt"
    )

    # Stack these positions into a 3x4 matrix.
    p: np.ndarray = np.column_stack([
        fiducials["0_0"],
        fiducials["0_1"],
        fiducials["0_2"],
        fiducials["0_3"]
    ])

    # TODO
    marker_positions: Dict[str, np.ndarray] = vicon.get_marker_positions("Registrar")

    q: np.ndarray = np.column_stack([
        marker_positions["0_0"],
        marker_positions["0_1"],
        marker_positions["0_2"],
        marker_positions["0_3"]
    ])

    print(p)
    print(q)

    # Estimate and return the rigid transformation between the two sets of points.
    transform: np.ndarray = GeometryUtil.estimate_rigid_transform(p, q)
    print(transform)
    # transform[0:3, 0:3] = np.array([
    #     [1.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0],
    #     [0.0, -1.0, 0.0]
    # ])
    return transform


def main() -> None:
    np.set_printoptions(suppress=True)

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

    # Connect to the Vicon system.
    with ViconInterface() as vicon:
        if vicon.get_frame():
            # Load in the scene mesh (if any).
            scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(
                "C:/spaint/build/bin/apps/spaintgui/meshes/TangoCapture-20210604-124834-cleaned.ply"
            )
            scene_mesh_o3d.transform(estimate_scene_transformation(vicon))
            scene_mesh: Optional[OpenGLTriMesh] = convert_trimesh_to_opengl(scene_mesh_o3d)

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
                            subject_from_world: Optional[np.ndarray] = vicon.get_segment_pose(subject, subject)

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

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
