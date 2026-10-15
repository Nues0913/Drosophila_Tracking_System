from setuptools import find_packages, setup

setup(
    name="flytrack_world",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/flytrack_world"]),
        ("share/flytrack_world", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="FlyTrack contributors",
    maintainer_email="maintainers@example.invalid",
    description="World and sensor corruption adapters for FlyTrack",
    license="MIT",
    entry_points={
        "console_scripts": [
            "odor_field = flytrack_world.odor_field:main",
            "target_mover = flytrack_world.target_mover:main",
            "scenario_manager = flytrack_world.scenario_manager:main",
            "sensor_corruptor = flytrack_world.sensor_corruptor:main",
        ]
    },
)
