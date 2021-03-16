import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.base import RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.comms.mapping import MappingServer
from smg.comms.skeletons import RemoteSkeletonDetector
from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.pyoctomap import *
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Skeleton, SkeletonRenderer, SkeletonUtil
from smg.utility import GeometryUtil


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--camera_mode", "-m", type=str, choices=("follow", "free"), default="free",
        help="the camera mode"
    )
    args: dict = vars(parser.parse_args())

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Octomap RGB-D Server")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Set up the octree drawer.
    drawer: OcTreeDrawer = OcTreeDrawer()
    drawer.set_color_mode(CM_COLOR_HEIGHT)

    # Create the octree.
    voxel_size: float = 0.05
    tree: OcTree = OcTree(voxel_size)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    with MappingServer(frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message) as server:
        with RemoteSkeletonDetector() as skeleton_detector:
            client_id: int = 0
            image_size: Optional[Tuple[int, int]] = None
            intrinsics: Optional[Tuple[float, float, float, float]] = None
            picker: Optional[OctomapPicker] = None
            receiver: RGBDFrameReceiver = RGBDFrameReceiver()
            skeletons: Optional[List[Skeleton]] = None
            tracker_w_t_c: Optional[np.ndarray] = None

            # Start the server.
            server.start()

            while True:
                # Process any PyGame events.
                for event in pygame.event.get():
                    # If the user wants to quit:
                    if event.type == pygame.QUIT:
                        # If the reconstruction process has actually started:
                        if tracker_w_t_c is not None:
                            # Save the current octree to disk.
                            print("Saving octree to remote_fusion_rgbd.bt")
                            tree.write_binary("remote_fusion_rgbd.bt")

                        # Shut down pygame, and forcibly exit the program.
                        pygame.quit()
                        # noinspection PyProtectedMember
                        os._exit(0)

                # If the server has a frame from the client that has not yet been processed:
                if server.has_frames_now(client_id):
                    # Get the camera parameters from the server.
                    height, width, _ = server.get_image_shapes(client_id)[0]
                    image_size = (width, height)
                    intrinsics = server.get_intrinsics(client_id)[0]

                    # Get the frame from the server.
                    server.get_frame(client_id, receiver)
                    tracker_w_t_c = receiver.get_pose()

                    skeleton_detector.set_calibration(image_size, intrinsics)
                    skeletons, people_mask = skeleton_detector.detect_skeletons(receiver.get_rgb_image(), tracker_w_t_c)

                    depth_image: np.ndarray = receiver.get_depth_image()
                    if people_mask is not None:
                        depth_image = SkeletonUtil.depopulate_depth_image(depth_image, people_mask)

                    # Use the depth image and pose to make an Octomap point cloud.
                    pcd: Pointcloud = OctomapUtil.make_point_cloud(
                        depth_image, tracker_w_t_c, intrinsics
                    )

                    # Fuse the point cloud into the octree.
                    start = timer()

                    origin: Vector3 = Vector3(0.0, 0.0, 0.0)
                    tree.insert_point_cloud(pcd, origin, discretize=True)

                    end = timer()
                    print(f"  - Time: {end - start}s")

                # Allow the user to control the camera.
                camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                # Clear the colour and depth buffers.
                glClearColor(1.0, 1.0, 1.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Once at least one frame has been received:
                if image_size is not None:
                    # Determine the viewing pose.
                    viewing_pose: np.ndarray = \
                        np.linalg.inv(tracker_w_t_c) if args["camera_mode"] == "follow" and tracker_w_t_c is not None \
                        else camera_controller.get_pose()

                    # Set the projection matrix.
                    with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                        GeometryUtil.rescale_intrinsics(intrinsics, image_size, window_size), *window_size
                    )):
                        # Set the model-view matrix.
                        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                            CameraPoseConverter.pose_to_modelview(viewing_pose)
                        )):
                            # Draw the octree.
                            OctomapUtil.draw_octree(tree, drawer)

                            # Render any detected 3D skeletons.
                            if skeletons is not None:
                                for skeleton in skeletons:
                                    SkeletonRenderer.render_skeleton(skeleton)

                            if skeletons is not None and len(skeletons) == 1:
                                if picker is None:
                                    # noinspection PyTypeChecker
                                    picker = OctomapPicker(tree, *image_size, intrinsics)

                                skeleton: Skeleton = skeletons[0]
                                lelbow: Skeleton.Keypoint = skeleton.keypoints.get("LElbow")
                                lwrist: Skeleton.Keypoint = skeleton.keypoints.get("LWrist")
                                picking_cam: SimpleCamera = SimpleCamera(
                                    lwrist.position, lwrist.position - lelbow.position, np.array([0.0, 1.0, 0.0])
                                )
                                picking_pose: np.ndarray = np.linalg.inv(
                                    CameraPoseConverter.camera_to_pose(picking_cam))
                                picking_image, picking_mask = picker.pick(picking_pose)
                                import cv2
                                cv2.imshow("Picking Image", picking_image)
                                cv2.waitKey(1)

                                width, height = image_size
                                if picking_mask[height // 2, width // 2] != 0:
                                    glColor3f(1, 0, 1)
                                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                                    OpenGLUtil.render_sphere(picking_image[height // 2, width // 2], 0.1, slices=10, stacks=10)
                                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                # Swap the front and back buffers.
                pygame.display.flip()


if __name__ == "__main__":
    main()
