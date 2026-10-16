from setuptools import setup

setup(
    name='delivery_metrics', version='0.1.0',
    packages=['delivery_metrics'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/delivery_metrics']),
        ('share/delivery_metrics', ['package.xml']),
    ],
    install_requires=['setuptools'],
    entry_points={'console_scripts': ['recorder = delivery_metrics.node:main']},
)
