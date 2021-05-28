from smg.vicon import ViconInterface


def main() -> None:
    with ViconInterface("169.254.185.150:801") as vicon:
        while True:
            if vicon.get_frame():
                print(vicon.get_subject_names())
                for subject in vicon.get_subject_names():
                    for segment in vicon.get_segment_names(subject):
                        print(subject, segment, vicon.get_segment_pose(subject, segment))

                print(f"=== Frame {vicon.get_frame_number()} ===")
                for name, pos in vicon.get_marker_positions("Wand").items():
                    print(name, pos)


if __name__ == "__main__":
    main()
