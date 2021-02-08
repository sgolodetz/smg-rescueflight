import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

from smg.imagesources import RGBFromRGBDImageSource, RGBImageSource
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera, OpenNIRGBDImageSource
from smg.pyorbslam2 import MonocularTracker
from smg.pyremode import DepthAssembler, DepthProcessor
from smg.rotory.drone_factory import DroneFactory
from smg.rotory.drone_rgb_image_source import DroneRGBImageSource


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--source_type", "-t", type=str, required=True, choices=("ardrone2", "kinect", "tello"),
        help="the input type"
    )
    args: dict = vars(parser.parse_args())

    image_source: Optional[RGBImageSource] = None
    try:
        # Construct the RGB image source.
        # FIXME: This is duplicate code - factor it out.
        source_type: str = args["source_type"]
        if source_type == "kinect":
            image_source = RGBFromRGBDImageSource(OpenNIRGBDImageSource(OpenNICamera(mirror_images=True)))
        else:
            kwargs: Dict[str, dict] = {
                "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
                "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
            }
            image_source = DroneRGBImageSource(DroneFactory.make_drone(source_type, **kwargs[source_type]))

        # Run the depth assembler.
        with MonocularTracker(
            settings_file=f"settings-{source_type}.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            image_size: Tuple[int, int] = image_source.get_image_size()
            intrinsics: Tuple[float, float, float, float] = image_source.get_intrinsics()
            depth_assembler: DepthAssembler = DepthAssembler(image_size, intrinsics)
            is_reference: bool = True

            reference_colour_image: Optional[np.ndarray] = None
            estimated_depth_image: Optional[np.ndarray] = None
            convergence_map: Optional[np.ndarray] = None

            _, ax = plt.subplots(2, 2)

            while True:
                # Get an RGB image from the camera.
                colour_image: np.ndarray = image_source.get_image()
                cv2.imshow("Image", colour_image)
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

                # Try to estimate the camera pose. If the tracker's not ready, or pose estimation fails, continue.
                if not tracker.is_ready():
                    continue
                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)
                if pose is None:
                    continue

                # Add the colour image and its estimated pose to the depth assembler.
                depth_assembler.put(colour_image, pose)

                # If this is the reference input, store the colour image for later, and ensure that future inputs
                # are not treated as the reference.
                if is_reference:
                    reference_colour_image = colour_image
                    is_reference = False

                # Try to get the latest version of the depth image that's being assembled.
                result = depth_assembler.get(blocking=False)
                if result is not None:
                    _, estimated_depth_image, _, converged_percentage, convergence_map = result
                    print(f"Converged: {converged_percentage}%")

                # Visualise the progress towards a suitable depth image. Move on once the user presses a key.
                ax[0, 0].clear()
                ax[0, 1].clear()
                ax[1, 0].clear()
                ax[1, 1].clear()
                ax[0, 0].imshow(reference_colour_image[:, :, [2, 1, 0]])
                if estimated_depth_image is not None:
                    ax[0, 1].imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
                ax[1, 0].imshow(colour_image[:, :, [2, 1, 0]])

                plt.draw()
                if plt.waitforbuttonpress(0.001):
                    break

            # If ORB-SLAM's not ready yet, forcibly terminate the whole process (this isn't graceful, but
            # if we don't do it then we may have to wait a very long time for it to finish initialising).
            if not tracker.is_ready():
                # noinspection PyProtectedMember
                os._exit(0)
    finally:
        # Terminate the image source once we've finished assembling a depth image.
        if image_source is not None:
            image_source.terminate()

        # Close any remaining OpenCV windows.
        cv2.destroyAllWindows()

    # Post-process the depth image, and visualise the result. Move on once the user presses a key.
    estimated_depth_image = DepthProcessor.postprocess_depth(estimated_depth_image, convergence_map, intrinsics)
    ax[1, 1].imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
    plt.draw()
    plt.waitforbuttonpress()

    # Destroy any PyPlot windows in existence.
    plt.close("all")

    # Visualise the keyframe as a coloured 3D point cloud.
    VisualisationUtil.visualise_rgbd_image(reference_colour_image, estimated_depth_image, intrinsics)


if __name__ == "__main__":
    main()

    # Make absolutely sure that the application exits - ORB-SLAM's viewer has a tendency to not close cleanly.
    # noinspection PyProtectedMember
    os._exit(0)
