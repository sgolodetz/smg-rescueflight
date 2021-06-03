import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

from smg.imagesources import RGBFromRGBDImageSource, RGBImageSource
from smg.openni import OpenNICamera, OpenNIRGBDImageSource
from smg.relocalisation import ArUcoPnPRelocaliser
from smg.rotory import DroneFactory, DroneRGBImageSource
from smg.utility import GeometryUtil, PoseUtil
from smg.vicon import ViconInterface


def estimate_source_pose(*, image: np.ndarray, intrinsics: Tuple[float, float, float, float],
                         registrar_subject_name: str, vicon: ViconInterface) -> Optional[np.ndarray]:
    # Try to determine the positions of the fiducials using the Vicon system. This can fail if the fiducials can't
    # currently be seen by the Vicon system, in which case an empty dictionary will be returned.
    fiducials: Dict[str, np.narray] = vicon.get_marker_positions(registrar_subject_name)

    # Set up a relocaliser that uses the known positions of the fiducials.
    relocaliser: ArUcoPnPRelocaliser = ArUcoPnPRelocaliser(fiducials)

    # Estimate the pose of the source using the relocaliser.
    return relocaliser.estimate_pose(image, intrinsics, draw_detections=False)


def main() -> None:
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--output_filename", "-o", type=str,
        help="the name of the file to which to save the relative transformation estimated (optional)"
    )
    parser.add_argument(
        "--registrar_subject_name", type=str, default="Registrar",
        help="the name of the registration board's Vicon subject"
    )
    parser.add_argument(
        "--source_subject_name", type=str, default="Tello",
        help="the name of the image source's Vicon subject"
    )
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

        # Connect to the Vicon system.
        with ViconInterface() as vicon:
            subject_from_source_estimates: List[np.ndarray] = []

            # Repeatedly:
            while True:
                # Get an image from the camera.
                image: np.ndarray = image_source.get_image()

                # Show the image. If the user presses the 'q' key, break out of the loop.
                cv2.imshow("Image", image)
                if cv2.waitKey(1) == ord('q'):
                    break

                # Try to get a frame of Vicon data. If it's available:
                if vicon.get_frame():
                    # Print out the frame number.
                    print(f"=== Frame {vicon.get_frame_number()} ===")

                    # Try to estimate the transformation from world space to the space of the image's Vicon subject,
                    # using the Vicon system.
                    source_subject_name: str = args["source_subject_name"]
                    subject_from_world_estimate: Optional[np.ndarray] = vicon.get_segment_pose(
                        source_subject_name, source_subject_name
                    )

                    # Try to estimate the relative transformation from the camera space of the image source to
                    # world space, using the ArUco marker and the Vicon system.
                    world_from_source_estimate: Optional[np.ndarray] = estimate_source_pose(
                        image=image,
                        intrinsics=image_source.get_intrinsics(),
                        registrar_subject_name=args["registrar_subject_name"],
                        vicon=vicon
                    )

                    # If both estimates were successfully obtained, multiply them to get an estimate of the relative
                    # transformation from the camera space of the image source to the space of its Vicon subject.
                    if subject_from_world_estimate is not None and world_from_source_estimate is not None:
                        subject_from_source_estimate = subject_from_world_estimate @ world_from_source_estimate
                        subject_from_source_estimates.append(subject_from_source_estimate)

                        print(subject_from_source_estimate)

            # Generate a more robust estimate of the relative transformation from the camera space of the image source
            # to the space of its Vicon subject.
            subject_from_source: np.ndarray = GeometryUtil.blend_rigid_transforms(subject_from_source_estimates)

            # Print out the estimate, and save it to disk if requested.
            print(subject_from_source)
            output_filename: Optional[str] = args.get("output_filename")
            if output_filename is not None:
                PoseUtil.save_pose(output_filename, subject_from_source)
    finally:
        # Destroy any remaining OpenCV windows.
        cv2.destroyAllWindows()

        # Terminate the image source.
        if image_source is not None:
            image_source.terminate()


if __name__ == "__main__":
    main()
