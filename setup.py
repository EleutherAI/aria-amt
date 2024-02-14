import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="aria-amt",
    py_modules=["amt"],
    version="0.0.1",
    description="",
    author="",
    packages=find_packages() + ["config"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "aria-amt=amt.run:main",
        ],
    },
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
)
