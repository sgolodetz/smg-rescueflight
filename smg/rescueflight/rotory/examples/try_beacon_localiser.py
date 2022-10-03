import numpy as np

from timeit import default_timer as timer
from typing import List, Tuple

from smg.rotory.beacons import BeaconLocaliser


def main() -> None:
    np.set_printoptions(suppress=True)

    # Specify the measured range to the beacon from several different locations.
    beacon_measurements: List[Tuple[np.ndarray, float]] = [
        (np.array([1, 0, 0]), 2.0),
        (np.array([-1, 0, 0]), 0.0),
        (np.array([0, 1, 0]), np.sqrt(2))
    ]

    # Try to localise the beacon.
    start = timer()
    beacon_pos: np.ndarray = BeaconLocaliser.try_localise_beacon(beacon_measurements)
    end = timer()

    # Print out the estimated position of the beacon, together with the time taken to estimate it.
    print(beacon_pos)
    print(f"{end - start}s")
    print()

    # Print out the error associated with each measurement for this estimate of the beacon's position.
    for receiver_pos, beacon_range in beacon_measurements:
        print(np.fabs(np.linalg.norm(receiver_pos - beacon_pos) - beacon_range))


if __name__ == "__main__":
    main()
