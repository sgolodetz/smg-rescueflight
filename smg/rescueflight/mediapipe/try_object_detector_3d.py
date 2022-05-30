import cv2
import numpy as np
import os

from typing import List, Tuple

from smg.mediapipe import ObjectDetector3D


def main() -> None:
    # TODO: Comment here.
    intrinsics: Tuple[float, float, float, float] = (
        1444.8084716796875, 1444.8084716796875, 962.52960205078125, 5919189453125
    )

    # TODO: Comment here.
    detector: ObjectDetector3D = ObjectDetector3D(image_size=(1920, 1440), intrinsics=intrinsics)

    # TODO: Comment here.
    image: np.ndarray = cv2.imread("frame_00948.jpg")

    # TODO: Comment here.
    objects: List[ObjectDetector3D.Object3D] = detector.detect_objects(image)


if __name__ == "__main__":
    main()
