from smg.utility import Cylinder, Sphere


def main() -> None:
    sphere: Sphere = Sphere(centre=[0, 0, 0], radius=1.0)
    print(sphere.classify_point([0.5, 0, 0]))
    print(sphere.classify_point([1, 0, 0]))
    print(sphere.classify_point([2, 0, 0]))
    print()

    cylinder: Cylinder = Cylinder(
        base_centre=[0, 0, 0], base_radius=1.0,
        top_centre=[0, 0, 10], top_radius=2.0
    )
    print(cylinder.classify_point([1, 0, 0]))
    print(cylinder.classify_point([1.49, 0, 5]))
    print(cylinder.classify_point([1.5, 0, 5]))
    print(cylinder.classify_point([1.51, 0, 5]))
    print(cylinder.classify_point([0, 0, -0.1]))
    print(cylinder.classify_point([0, 0, 0]))
    print(cylinder.classify_point([0, 0, 0.1]))
    print(cylinder.classify_point([0, 0, 9.9]))
    print(cylinder.classify_point([0, 0, 10]))
    print(cylinder.classify_point([0, 0, 10.1]))


if __name__ == "__main__":
    main()
