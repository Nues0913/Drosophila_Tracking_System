from glob import glob

from setuptools import find_packages, setup

package_name = "flytrack_eval"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "launch_testing"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch_testing", glob("launch_testing/*.py")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="FlyTrack maintainers",
    maintainer_email="maintainers@example.invalid",
    description="Read-only evaluation and rosbag recording",
    license="MIT",
    entry_points={
        "console_scripts": [
            "evaluator_node = flytrack_eval.evaluator_node:main",
            "bag_recorder_node = flytrack_eval.bag_recorder_node:main",
        ]
    },
)
