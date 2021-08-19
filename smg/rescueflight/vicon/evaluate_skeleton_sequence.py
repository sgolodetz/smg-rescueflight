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
from smg.utility import MarkerUtil, PoseUtil
from smg.vicon import OfflineViconInterface, ViconSkeletonDetector, ViconUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--detector_type", "-t", type=str, default="lcrnet", choices=("lcrnet", "xnect"),
        help="the skeleton detector whose (pre-saved) skeletons are to be evaluated"
    )
    parser.add_argument(
        "--sequence_dir", "-s", type=str, required=True,
        help="the name of the directory containing the ground-truth Vicon sequence"
    )
    args: dict = vars(parser.parse_args())

    detector_type: str = args["detector_type"]
    sequence_dir: str = args["sequence_dir"]

    # Try to load in the transformation from world space to Vicon space.
    vicon_from_world_filename: str = os.path.join(sequence_dir, "reconstruction", "vicon_from_world.txt")
    if os.path.exists(vicon_from_world_filename):
        vicon_from_world: np.ndarray = PoseUtil.load_pose(vicon_from_world_filename)
    else:
        raise RuntimeError(f"'{vicon_from_world_filename}' does not exist")

    # Load in the reconstructed scene mesh and transform it into Vicon space.
    mesh_filename: str = os.path.join(sequence_dir, "reconstruction", "mesh.ply")
    scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_filename)
    scene_mesh_o3d.transform(vicon_from_world)
    scene_mesh: OpenGLTriMesh = MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d)

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
        SimpleCamera([0, 0, 0], [0, 1, 0], [0, 0, 1]), canonical_angular_speed=0.05,
        canonical_linear_speed=0.1
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
        evaluate: bool = False
        pause: bool = True
        process_next: bool = True
        vicon_from_gt: Optional[np.ndarray] = None

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
                    # noinspection PyProtectedMember
                    os._exit(0)

            # If we're ready to do so, process the next frame. Also record whether we processed a frame or not.
            processed_frame: bool = False
            if process_next:
                vicon.get_frame()
                processed_frame = True
                process_next = not pause

            # Get the frame number of the current Vicon frame, and print it out.
            frame_number: int = vicon.get_frame_number()

            print("===")
            print(f"Frame {frame_number}")

            # Turn evaluation on or off based on the existence or absence of files in the evalcontrol sub-directory.
            # This is used to avoid penalising missed detections when the person cannot be seen from the camera.
            # FIXME: This works ok for single-person scenes, but for multi-person scenes we'll need to figure out
            #        which people can be seen in a particular image and which can't.
            if os.path.exists(os.path.join(sequence_dir, "evalcontrol", f"{frame_number}-on.txt")):
                evaluate = True
            if os.path.exists(os.path.join(sequence_dir, "evalcontrol", f"{frame_number}-off.txt")):
                evaluate = False

            # If the transformation from ground-truth space to Vicon space hasn't yet been calculated:
            if vicon_from_gt is None:
                from smg.utility import FiducialUtil

                # Try to calculate it now.
                fiducials_filename: str = os.path.join(sequence_dir, "gt", "fiducials.txt")
                gt_marker_positions: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(fiducials_filename)

                # Look up the Vicon coordinate system positions of the all of the Vicon markers that can currently
                # be seen by the Vicon system, hopefully including ones for the ArUco marker corners.
                vicon_marker_positions: Dict[str, np.ndarray] = vicon.get_marker_positions("Registrar")

                vicon_from_gt = MarkerUtil.estimate_space_to_space_transform(
                    gt_marker_positions, vicon_marker_positions
                )

                # If this fails, raise an exception.
                if vicon_from_gt is None:
                    raise RuntimeError("Couldn't calculate ground-truth to Vicon space transform - check the markers")

            ###
            # # Load in the reconstructed scene mesh and transform it into Vicon space.
            # mesh_filename: str = os.path.join(sequence_dir, "gt", "mesh.ply")
            # scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_filename)
            # scene_mesh_o3d.transform(vicon_from_gt)
            # scene_mesh: OpenGLTriMesh = MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d)
            ###

            # Get the ground-truth and (previously) detected skeletons.
            gt_skeletons: Dict[str, Skeleton3D] = gt_skeleton_detector.detect_skeletons()

            detected_skeletons: Optional[List[Skeleton3D]] = SkeletonUtil.try_load_skeletons(
                os.path.join(sequence_dir, detector_type, f"{frame_number}.skeletons.txt")
            )

            # Print out the number of skeleton matches we've established, for debugging purposes.
            print(f"Matched Skeleton Count: {len(matched_skeletons)}")

            # If we've just processed a new Vicon frame:
            if processed_frame:
                # If the frame only contains a single ground-truth skeleton, and evaluation is enabled:
                if len(gt_skeletons) == 1 and evaluate:
                    # Get the ground-truth skeleton.
                    gt_skeleton: Skeleton3D = list(gt_skeletons.values())[0]

                    # If a single skeleton was detected in this frame, transform it into Vicon space and match it
                    # with the ground-truth one. Otherwise, record that the ground-truth skeleton has no match.
                    if detected_skeletons is not None and len(detected_skeletons) == 1:
                        detected_skeleton: Skeleton3D = detected_skeletons[0]
                        detected_skeleton = detected_skeleton.transform(vicon_from_world)
                        matched_skeletons.append([(gt_skeleton, detected_skeleton)])
                    else:
                        matched_skeletons.append([(gt_skeleton, None)])

                # If we've previously established at least one skeleton match:
                if len(matched_skeletons) > 0:
                    # Calculate the evaluation metrics for all the matches we've seen so far, and print them out.
                    per_joint_error_table: np.ndarray = skeleton_evaluator.make_per_joint_error_table(matched_skeletons)
                    print(per_joint_error_table)
                    mpjpes: Dict[str, float] = skeleton_evaluator.calculate_mpjpes(per_joint_error_table)
                    print(mpjpes)
                    correct_keypoint_table: np.ndarray = SkeletonEvaluator.make_correct_keypoint_table(
                        per_joint_error_table, threshold=0.5
                    )
                    print(correct_keypoint_table)
                    pcks: Dict[str, float] = skeleton_evaluator.calculate_pcks(correct_keypoint_table)
                    print(pcks)

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
                    OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                    # Render the reconstructed scene.
                    scene_mesh.render()

                    # Render the ArUco marker (this will be at the origin in ArUco space).
                    # with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(aruco_from_vicon)):
                    if all(key in vicon_marker_positions for key in ["0_0", "0_1", "0_2", "0_3"]):
                        glBegin(GL_QUADS)
                        glColor3f(0, 1, 0)
                        glVertex3f(*vicon_marker_positions["0_0"])
                        glVertex3f(*vicon_marker_positions["0_1"])
                        glVertex3f(*vicon_marker_positions["0_2"])
                        glVertex3f(*vicon_marker_positions["0_3"])
                        glEnd()

                    # Render the 3D skeletons in their Vicon-space locations.
                    with SkeletonRenderer.default_lighting_context():
                        # with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(aruco_from_vicon)):
                        for _, skeleton in gt_skeletons.items():
                            SkeletonRenderer.render_skeleton(skeleton)

                        if detected_skeletons is not None:
                            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(vicon_from_world)):
                                for skeleton in detected_skeletons:
                                    SkeletonRenderer.render_skeleton(skeleton)

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
