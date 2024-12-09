'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    setup.py
'''
from setuptools import setup, find_packages
from typing import List
import os


def get_path(*filepath) -> str:
    ROOT_DIR = os.path.dirname(__file__)
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements():
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements


setup(
    name="dashinfer-vlm",
    version=os.getenv("VLM_RELEASE_VERSION", "2.3.0"),
    author="DashInfer team",
    author_email="Dash-Infer@alibabacloud.com",
    description="DashInfer VLM is a native inference engine for Pre-trained Vision Language Models (VLMs) developed by Tongyi Laboratory.",
    packages=find_packages(),
    install_requires=get_requirements(),
    classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.10',
    ],
    license="Apache License 2.0",
    entry_points={
        "console_scripts": [
            "dashinfer_vlm_serve = dashinfer_vlm.api_server.server:main",
        ],
    },
)
