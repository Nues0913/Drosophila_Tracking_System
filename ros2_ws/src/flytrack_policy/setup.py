from setuptools import find_packages, setup

package_name = "flytrack_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "torch"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="FlyTrack contributors",
    maintainer_email="maintainers@example.invalid",
    description="Sensory-only connectome tracking policy with simulation-time watchdog",
    license="MIT",
    entry_points={"console_scripts": ["connectome_policy_node = flytrack_policy.node:main"]},
)
