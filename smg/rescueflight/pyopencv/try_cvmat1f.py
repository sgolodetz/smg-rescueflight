import numpy as np

from smg.pyopencv import CVMat1f


def main():
    mat: CVMat1f = CVMat1f.zeros(2, 3)
    arr: np.ndarray = np.array(mat, copy=False)
    print(arr)


if __name__ == "__main__":
    main()
