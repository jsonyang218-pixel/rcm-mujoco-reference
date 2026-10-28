#!/usr/bin/env python3
"""Install the reusable Python modules for the ``ros_msg`` catkin package."""

from setuptools import find_packages, setup


setup(
    name="ros_msg",
    version="0.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
