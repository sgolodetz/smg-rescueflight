from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="smg-rescueflight",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Top-level scripts related to our application domain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-rescueflight",
    packages=find_packages(include=["smg.rescueflight", "smg.rescueflight.*"]),
    include_package_data=True,
    install_requires=[
        "smg-comms",
        "smg-dvmvs",
        "smg-imagesources",
        "smg-joysticks",
        "smg-mapping",
        "smg-meshing",
        "smg-mvdepthnet",
        "smg-navigation",
        "smg-open3d",
        "smg-opengl",
        "smg-pyleap",
        "smg-pyoctomap",
        "smg-pyopencv",
        "smg-pyorbslam2",
        "smg-relocalisation",
        "smg-rigging",
        "smg-rotorsim",
        "smg-rotory",
        "smg-skeletons",
        "smg-smplx",
        "smg-utility"
    ],
    extras_require={
        "all": [
            "smg-openni",
            "smg-pyopenpose",
            "smg-pyorbslam3",
            "smg-pyremode",
            "smg-robotdepot",
            "smg-vicon"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7.*',
)
