import numpy as np

from timeit import default_timer as timer
from typing import List, Tuple

from smg.rotory.beacons import BeaconLocaliser


def main() -> None:
    np.set_printoptions(suppress=True)

    beacon_measurements: List[Tuple[np.ndarray, float]] = [
        (np.array([1, 0, 0]), 2.0),
        (np.array([-1, 0, 0]), 0.0),
        (np.array([0, 1, 0]), np.sqrt(2))
    ]

    start = timer()
    beacon_pos: np.ndarray = BeaconLocaliser.try_localise_beacon(beacon_measurements)
    end = timer()

    print(beacon_pos)
    print(f"{end - start}s")
    print()

    for receiver_pos, beacon_range in beacon_measurements:
        print(np.fabs(np.linalg.norm(receiver_pos - beacon_pos) - beacon_range))


if __name__ == "__main__":
    main()
