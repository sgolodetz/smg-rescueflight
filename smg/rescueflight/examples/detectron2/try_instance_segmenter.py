import cv2
import numpy as np

from detectron2.structures import Instances
from typing import List, Optional

from smg.detectron2 import InstanceSegmenter


def main() -> None:
    # Construct the instance segmenter.
    segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()

    # For each image in the sequence:
    frame_idx: int = 0
    while True:
        image: Optional[np.ndarray] = cv2.imread(f"C:/smglib/smg-mapping/output-metric/frame-{frame_idx:06d}.color.png")
        if image is None:
            break

        # Segment the image.
        raw_instances: Instances = segmenter.segment_raw(image)
        instances: List[InstanceSegmenter.Instance] = segmenter.parse_raw_instances(raw_instances)

        # Show both the image and the segmentation result.
        cv2.imshow("Image", image)
        cv2.imshow("Output", segmenter.draw_raw_instances(raw_instances, image))

        # Show the individual instance masks.
        for i in range(len(instances)):
            instance: InstanceSegmenter.Instance = instances[i]
            cv2.imshow(f"Mask {i}", instance.mask)

        c: int = cv2.waitKey(1)
        if c == ord('q'):
            break

        frame_idx += 1


if __name__ == "__main__":
    main()
