import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

from smg.meshing import MeshUtil
from smg.opengl import OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Skeleton3D, SkeletonEvaluator, SkeletonRenderer, SkeletonUtil


def get_frame_index(filename: str) -> int:
    """
    Get the frame index corresponding to a file containing skeleton data.

    .. note::
        The files are named <frame index>.skeletons.txt, so we can get the frame indices directly from the file names.

    :param filename:    The name of a file containing skeleton data.
    :return:            The corresponding frame index.
    """
    frame_idx, _, _ = filename.split(".")
    return int(frame_idx)


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--detector_type", "-t", type=str, default="lcrnet", choices=("gt", "lcrnet", "xnect"),
        help="the skeleton detector whose (pre-saved) skeletons are to be evaluated"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the name of the directory containing the sequence"
    )
    args: dict = vars(parser.parse_args())

    detector_type: str = args["detector_type"]
    sequence_dir: str = args["sequence_dir"]

    # Try to load in the ground-truth mesh.
    scene_mesh: Optional[OpenGLTriMesh] = None
    mesh_filename: str = os.path.join(sequence_dir, "recon", "gt_gt.ply")
    if os.path.exists(mesh_filename):
        scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_filename)
        scene_mesh = MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Skeleton Sequence Evaluator")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05,
        canonical_linear_speed=0.1
    )

    # Construct the skeleton evaluator and initialise the list of matched skeletons.
    skeleton_evaluator: SkeletonEvaluator = SkeletonEvaluator.make_default()
    matched_skeletons: List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]] = []

    # Determine the list of skeleton filenames to use.
    skeleton_filenames: List[str] = [
        f for f in os.listdir(os.path.join(sequence_dir, "people", "gt")) if f.endswith(".skeletons.txt")
    ]

    skeleton_filenames = sorted(skeleton_filenames, key=get_frame_index)

    # Initialise a few variables.
    detected_skeletons: Optional[List[Skeleton3D]] = None
    gt_skeletons: Optional[List[Skeleton3D]] = None
    pause: bool = True
    process_next: bool = False

    # Until we reach the end of the sequence:
    i: int = 0
    while i < len(skeleton_filenames):
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
                # If the user wants us to quit, shut down pygame.
                pygame.quit()

                # Then forcibly terminate the whole process.
                # noinspection PyProtectedMember
                os._exit(0)

        # If we're ready to do so, process the next frame.
        if process_next:
            # Print out the frame index.
            print("===")
            print(f"Frame {get_frame_index(skeleton_filenames[i])}")

            # Get the ground-truth and (previously) detected skeletons for the frame.
            gt_skeletons = SkeletonUtil.try_load_skeletons(
                os.path.join(sequence_dir, "people", "gt", skeleton_filenames[i])
            )

            detected_skeletons = SkeletonUtil.try_load_skeletons(
                os.path.join(sequence_dir, "people", detector_type, skeleton_filenames[i])
            )

            # BEGIN TEMPORARY
            # This is for variety, since I can't run LCR-Net on the desktop in the lab.
            if detected_skeletons is not None:
                for j in range(len(detected_skeletons)):
                    detected_skeletons[j] = detected_skeletons[j].transform(np.array([
                        [1.0, 0.0, 0.0, np.random.normal(0.15, 0.05)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ]))
            # END TEMPORARY

            # If the frame contains a single ground-truth skeleton:
            if gt_skeletons is not None and len(gt_skeletons) == 1:
                # Get the ground-truth skeleton.
                gt_skeleton: Skeleton3D = gt_skeletons[0]

                # If a single skeleton was detected in this frame, match it with the ground-truth one. Otherwise,
                # record that the ground-truth skeleton has no match.
                if detected_skeletons is not None and len(detected_skeletons) == 1:
                    detected_skeleton: Skeleton3D = detected_skeletons[0]
                    matched_skeletons.append([(gt_skeleton, detected_skeleton)])
                else:
                    matched_skeletons.append([(gt_skeleton, None)])

            # If we've previously established at least one skeleton match:
            if len(matched_skeletons) > 0:
                # Calculate the evaluation metrics for all the matches we've seen so far, and print them out.
                per_joint_error_table: np.ndarray = skeleton_evaluator.make_per_joint_error_table(
                    matched_skeletons)
                print(per_joint_error_table)
                mpjpes: Dict[str, float] = skeleton_evaluator.calculate_mpjpes(per_joint_error_table)
                print(mpjpes)
                correct_keypoint_table: np.ndarray = SkeletonEvaluator.make_correct_keypoint_table(
                    per_joint_error_table, threshold=0.15
                )
                print(correct_keypoint_table)
                pcks: Dict[str, float] = skeleton_evaluator.calculate_pcks(correct_keypoint_table)
                print(pcks)

            # Advance to the next frame.
            i += 1
            process_next = not pause

        # Allow the user to control the camera.
        camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

        # Clear the colour and depth buffers.
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix.
        with OpenGLMatrixContext(
            GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix((500.0, 500.0, 320.0, 240.0), 640, 480)
        ):
            # Set the model-view matrix.
            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
            )):
                # Render a voxel grid.
                glColor3f(0.0, 0.0, 0.0)
                OpenGLUtil.render_voxel_grid([-3, -2, -3], [3, 0, 3], [1, 1, 1], dotted=True)

                # If the scene's available, render it.
                if scene_mesh is not None:
                    scene_mesh.render()

                # If 3D skeletons are available for this frame, render them.
                with SkeletonRenderer.default_lighting_context():
                    if gt_skeletons is not None:
                        for skeleton in gt_skeletons:
                            SkeletonRenderer.render_skeleton(skeleton)

                    if detected_skeletons is not None:
                        for skeleton in detected_skeletons:
                            SkeletonRenderer.render_skeleton(skeleton)

        # Swap the front and back buffers.
        pygame.display.flip()


if __name__ == "__main__":
    main()
