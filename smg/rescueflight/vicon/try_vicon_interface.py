import numpy as np

from smg.vicon import LiveViconInterface


def main() -> None:
    np.set_printoptions(suppress=True)

    with LiveViconInterface() as vicon:
        while True:
            if vicon.get_frame():
                print(f"=== Frame {vicon.get_frame_number()} ===")

                for name, pos in vicon.get_marker_positions("Wand").items():
                    print(name, pos)

                print(vicon.get_subject_names())
                for subject in vicon.get_subject_names():
                    for segment in vicon.get_segment_names(subject):
                        print(subject, segment, vicon.get_segment_global_pose(subject, segment))


if __name__ == "__main__":
    main()
