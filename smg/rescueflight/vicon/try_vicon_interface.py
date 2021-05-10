import numpy as np

from typing import Dict, Optional

from smg.vicon import ViconInterface


def main() -> None:
    with ViconInterface("169.254.185.150:801") as vicon:
        while True:
            marker_positions: Optional[Dict[str, np.ndarray]] = vicon.try_get_marker_positions("Madhu")
            if marker_positions is not None:
                for name, pos in marker_positions.items():
                    print(name, pos)
                print("===")


if __name__ == "__main__":
    main()
