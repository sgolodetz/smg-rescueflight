import matplotlib.pyplot as plt

from smg.openni.openni_camera import OpenNICamera


def main():
    with OpenNICamera(mirror_images=True) as camera:
        print(f"Colour Camera: Dims={camera.get_colour_size()}, Intrinsics={camera.get_colour_intrinsics()}")
        print(f"Depth Camera: Dims={camera.get_depth_size()}, Intrinsics={camera.get_depth_intrinsics()}")

        _, ax = plt.subplots(1, 2)

        while True:
            colour_image, depth_image = camera.get_images()

            ax[0].clear()
            ax[1].clear()
            ax[0].imshow(colour_image)
            ax[1].imshow(depth_image)

            plt.draw()
            plt.waitforbuttonpress(0.001)


if __name__ == "__main__":
    main()
