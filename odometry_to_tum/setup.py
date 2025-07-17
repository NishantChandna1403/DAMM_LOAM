from setuptools import setup

package_name = 'odometry_to_tum'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Node to save odometry as TUM and KITTI trajectory formats',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odometry_to_tum = odometry_to_tum.odometry_to_tum:main',
            'odometry_to_kitti = odometry_to_tum.odometry_to_kitti:main'
        ],
    },
)
