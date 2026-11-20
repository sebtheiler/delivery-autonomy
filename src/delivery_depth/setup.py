from setuptools import setup
from glob import glob

setup(
    name='delivery_depth', version='0.1.0', packages=['delivery_depth'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/delivery_depth']),
        ('share/delivery_depth', ['package.xml']),
        ('share/delivery_depth/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    entry_points={'console_scripts': [
        'processor = delivery_depth.node:main',
        'cloud_processor = delivery_depth.cloud_node:main',
    ]},
)
