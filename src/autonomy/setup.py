from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autonomy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sebtheiler',
    maintainer_email='25444757+sebtheiler@users.noreply.github.com',
    description='State estimation, global planning, and MPPI control for the delivery robot',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'state_estimation = autonomy.state_estimation:main',
            'global_planner = autonomy.global_planner:main',
            'controller = autonomy.controller:main',
        ],
    },
)
