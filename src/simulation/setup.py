from setuptools import find_packages, setup
import os
from glob import glob

package_name = "simulation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (
            os.path.join("share", package_name, "models/washu"),
            glob("models/washu/*.sdf") + glob("models/washu/*.config"),
        ),  # Grab sdf and config
        (
            os.path.join("share", package_name, "models/washu/meshes"),
            glob("models/washu/meshes/*"),
        ),  # Grab obj and mtl
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sebtheiler",
    maintainer_email="25444757+sebtheiler@users.noreply.github.com",
    description="Gazebo world, robot model, and ground truth publisher for the delivery robot",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            'ground_truth_node = simulation.ground_truth_node:main',
        ],
    },
)
