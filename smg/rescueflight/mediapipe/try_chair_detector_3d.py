import os

import cv2
import numpy as np
import open3d as o3d

from typing import Any, Dict, List, Optional, Tuple

from smg.mediapipe import ChairDetector3D
from smg.open3d import VisualisationUtil
from smg.utility import SequenceUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    # Specify the camera intrinsics.
    intrinsics: Tuple[float, float, float, float] = (
        1444.8084716796875, 1444.8084716796875, 962.52960205078125, 726.5919189453125
        # 191.9573516845703, 191.9573516845703, 128.18414306640625, 96.97889709472656
    )

    # Construct the chair detector.
    detector: ChairDetector3D = ChairDetector3D(
        debug=True, image_size=(1920, 1440), intrinsics=intrinsics, min_detection_confidence=0.6
    )
    # detector: ChairDetector3D = ChairDetector3D(debug=True, image_size=(256, 192), intrinsics=intrinsics)

    # Load in the scene mesh.
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("C:/iphonescans/2022_05_17_14_37_57-aligned/mesh.ply")

    # Start setting up the 3D visualisation.
    to_visualise: List[o3d.geometry.Geometry] = [mesh]

    # Get the filenames of the full-resolution images.
    fullres_image_filenames: List[str] = sorted([
        f for f in os.listdir("C:/iphonescans/2022_05_17_14_37_57") if f.endswith(".jpg")
    ])

    # Initialise some variables.
    frame_idx: int = 0
    # noinspection PyUnusedLocal
    world_from_camera: np.ndarray = np.eye(4)

    # Repeatedly:
    while True:
        # Try to read in the next frame in the sequence.
        frame: Optional[Dict[str, Any]] = SequenceUtil.try_load_rgbd_frame(
            frame_idx, "C:/iphonescans/2022_05_17_14_37_57-aligned"
        )

        # If that fails, exit.
        if frame is None:
            break

        # Read in the full-resolution version of the colour image for the frame.
        image: np.ndarray = cv2.imread(
            os.path.join("C:/iphonescans/2022_05_17_14_37_57", fullres_image_filenames[frame_idx])
        )

        # Extract the camera pose from the frame.
        world_from_camera = frame["world_from_camera"]

        # Print out and then increment the frame index.
        print(frame_idx)
        frame_idx += 1

        # Try to detect any chairs that can be seen in the current frame.
        detected_chairs: List[ChairDetector3D.Chair] = detector.detect_chairs(image, world_from_camera)

        # Specify the colours to give the detected chairs' 3D landmarks.
        landmark_colours: List[Tuple[float, float, float]] = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0.5, 0), (1, 0, 0.5), (0, 0.5, 1)
        ]

        # Add the Open3D geometries for the detected chairs to the 3D visualisation.
        for chair in detected_chairs:
            to_visualise += chair.make_o3d_geometries(box_colour=(0, 1, 0), landmark_colours=landmark_colours)

    # Run the 3D visualisation.
    VisualisationUtil.visualise_geometries(to_visualise)


if __name__ == "__main__":
    main()
