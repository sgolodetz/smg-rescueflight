import numpy as np

from typing import List

from smg.utility import Cylinder, ShapeUtil, Sphere


def main() -> None:
    sphere: Sphere = Sphere(centre=[0, 0, 0], radius=1.0)
    print(sphere.mins(), sphere.maxs())
    print(sphere.classify_point([0.5, 0, 0]))
    print(sphere.classify_point([1, 0, 0]))
    print(sphere.classify_point([2, 0, 0]))
    print()

    cylinder: Cylinder = Cylinder(
        base_centre=[0, 0, 0], base_radius=1.0,
        top_centre=[0, 0, 10], top_radius=2.0
    )
    print(cylinder.mins(), cylinder.maxs())
    print(cylinder.classify_point([0.99, 0, 0]))
    print(cylinder.classify_point([1, 0, 0]))
    print(cylinder.classify_point([1.01, 0, 0]))
    print(cylinder.classify_point([1.49, 0, 5]))
    print(cylinder.classify_point([1.5, 0, 5]))
    print(cylinder.classify_point([1.51, 0, 5]))
    print(cylinder.classify_point([0, 0, -0.1]))
    print(cylinder.classify_point([0, 0, 0]))
    print(cylinder.classify_point([0, 0, 0.1]))
    print(cylinder.classify_point([0, 0, 9.9]))
    print(cylinder.classify_point([0, 0, 10]))
    print(cylinder.classify_point([0, 0, 10.1]))
    print()

    from pprint import pprint
    from timeit import default_timer as timer
    start = timer()
    voxels: List[np.ndarray] = ShapeUtil.rasterise_shapes([sphere, cylinder], 5.0)
    end = timer()
    pprint(voxels)
    print(f"Time: {end - start}s")


if __name__ == "__main__":
    main()
