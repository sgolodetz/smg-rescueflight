import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame

from argparse import ArgumentParser
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Dict, List, Optional, Set, Tuple

from smg.meshing import MeshUtil
from smg.opengl import OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Skeleton3D, SkeletonEvaluator, SkeletonRenderer, SkeletonUtil
from smg.utility import PoseUtil
from smg.vicon import OfflineViconInterface, ViconSkeletonDetector, ViconUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--batch", action="store_true",
        help="whether to run in batch mode"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="whether to print out per-frame metrics for debugging purposes"
    )
    parser.add_argument(
        "--detector_tag", "-t", type=str, default="lcrnet", choices=("lcrnet", "xnect"),
        help="the tag of the skeleton detector whose (pre-saved) skeletons are to be evaluated"
    )
    parser.add_argument(
        "--mesh_type", "-m", type=str, default="reconstruction", choices=("gt", "reconstruction"),
        help="which mesh to show for the scene (the ground truth or the reconstruction)"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the name of the directory containing the ground-truth Vicon sequence"
    )
    args: dict = vars(parser.parse_args())

    batch: bool = args["batch"]
    debug: bool = args["debug"]
    detector_tag: str = args["detector_tag"]
    mesh_type: str = args["mesh_type"]
    sequence_dir: str = args["sequence_dir"]

    # Try to load in the transformation from world space to Vicon space.
    vicon_from_world_filename: str = os.path.join(sequence_dir, "reconstruction", "vicon_from_world.txt")
    if os.path.exists(vicon_from_world_filename):
        vicon_from_world: np.ndarray = PoseUtil.load_pose(vicon_from_world_filename)
    else:
        raise RuntimeError(f"'{vicon_from_world_filename}' does not exist")

    # Try to load in the scene mesh (if this exists, it will already be in Vicon space).
    mesh_filename: str = os.path.join(sequence_dir, mesh_type, "vicon_mesh.ply")
    # noinspection PyUnresolvedReferences
    scene_mesh: Optional[OpenGLTriMesh] = \
        MeshUtil.convert_trimesh_to_opengl(o3d.io.read_triangle_mesh(mesh_filename)) \
        if os.path.exists(mesh_filename) else None

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
        SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the skeleton evaluator and initialise the list of matched skeletons.
    skeleton_evaluator: SkeletonEvaluator = SkeletonEvaluator.make_default()
    matched_skeletons: List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]] = []

    # Connect to the Vicon interface.
    with OfflineViconInterface(folder=sequence_dir) as vicon:
        # Construct the ground-truth skeleton detector.
        gt_skeleton_detector: ViconSkeletonDetector = ViconSkeletonDetector(
            vicon, is_person=ViconUtil.is_person, use_vicon_poses=True
        )

        # Initialise a few variables.
        all_subjects_visible: bool = False
        detected_skeletons: Optional[List[Skeleton3D]] = None
        gt_skeletons: Dict[str, Skeleton3D] = {}
        pause: bool = not batch
        process_next: bool = True
        visible_subjects: Set[str] = set()

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
                    # If the user wants us to quit, shut down pygame.
                    pygame.quit()

                    # Then forcibly terminate the whole process.
                    # noinspection PyProtectedMember, PyUnresolvedReferences
                    os._exit(0)

            # If we're ready to do so, process the next frame.
            if process_next:
                # If we run out of frames to process, break out of the loop.
                if not vicon.get_frame():
                    break

                # TODO: Move to the bottom.
                process_next = not pause

                # Get the frame number of the current Vicon frame.
                # FIXME: This can be None, and we should be checking for it.
                frame_number: int = vicon.get_frame_number()

                # Get the ground-truth and (previously) detected skeletons.
                gt_skeletons = gt_skeleton_detector.detect_skeletons()

                detected_skeletons = SkeletonUtil.try_load_skeletons(
                    os.path.join(sequence_dir, "people", detector_tag, f"{frame_number}.skeletons.txt")
                )

                # Transform the detected skeletons (if any) into Vicon space.
                if detected_skeletons is not None:
                    detected_skeletons = [skeleton.transform(vicon_from_world) for skeleton in detected_skeletons]

                # Filter out any ground-truth skeletons that cannot currently be seen from the camera.
                # FIXME: Do this *after* turning evaluation on/off.
                visible_gt_skeletons: Dict[str, Skeleton3D] = {
                    subject: skeleton for subject, skeleton in gt_skeletons.items()
                    if all_subjects_visible or subject in visible_subjects
                }

                # Turn evaluation on or off, either globally or for individual subjects. This is used to
                # avoid penalising missed detections when a person cannot be seen from the camera.
                if os.path.exists(os.path.join(sequence_dir, "evalcontrol", f"{frame_number}-on.txt")):
                    all_subjects_visible = True
                if os.path.exists(os.path.join(sequence_dir, "evalcontrol", f"{frame_number}-off.txt")):
                    all_subjects_visible = False

                filename: str = os.path.join(sequence_dir, "evalcontrol", f"{frame_number}.txt")
                if os.path.exists(filename):
                    with open(filename) as f:
                        lines: List[str] = [line.strip() for line in f.readlines() if line.strip() != ""]
                        for line in lines:
                            subject, onoff = line.split(" ")
                            if onoff == "on" and subject not in visible_subjects:
                                visible_subjects.add(subject)
                            elif onoff == "off" and subject in visible_subjects:
                                visible_subjects.remove(subject)

                evaluate: bool = all_subjects_visible or len(visible_subjects) != 0

                # If evaluation is enabled and any skeletons have been detected:
                if evaluate and detected_skeletons is not None:
                    # Match the detected skeletons with the visible ground-truth ones.
                    new_matches: List[Tuple[Skeleton3D, Optional[Skeleton3D]]] = \
                        SkeletonUtil.match_detections_with_ground_truth(
                            detected_skeletons=detected_skeletons, gt_skeletons=list(visible_gt_skeletons.values())
                        )

                    # Add the matches to the overall list.
                    matched_skeletons.append(new_matches)

                # If we're debugging, calculate and print the evaluation metrics for all the matches we've seen so far.
                if debug:
                    print("===")
                    print(f"Frame {frame_number}")
                    print("Visible Subjects:", ("All" if all_subjects_visible else visible_subjects))
                    print(f"Evaluated Frames: {len(matched_skeletons)}")
                    skeleton_evaluator.print_metrics(matched_skeletons)

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
                    OpenGLUtil.render_voxel_grid([-3, -7, 0], [3, 7, 2], [1, 1, 1], dotted=True)

                    # Render the reconstructed scene.
                    if scene_mesh is not None:
                        scene_mesh.render()

                    # Render the 3D skeletons in their Vicon-space locations.
                    with SkeletonRenderer.default_lighting_context():
                        for subject, skeleton in gt_skeletons.items():
                            if all_subjects_visible or subject in visible_subjects:
                                SkeletonRenderer.render_skeleton(skeleton, uniform_colour=(0, 1, 0))
                            else:
                                SkeletonRenderer.render_skeleton(skeleton, uniform_colour=(0.5, 1, 1))

                        if detected_skeletons is not None:
                            for skeleton in detected_skeletons:
                                SkeletonRenderer.render_skeleton(skeleton)

            # Swap the front and back buffers.
            pygame.display.flip()

    # If we're debugging, print a blank line before the summary metrics.
    if debug:
        print()

    # Calculate and print out the summary metrics.
    skeleton_evaluator.print_metrics(matched_skeletons)


if __name__ == "__main__":
    main()
