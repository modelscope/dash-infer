#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    ArgParser.py
#
import os
import json
import argparse


def save_config_as_json(config, file_path):
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)


def get_config_from_json(file_path):
    config = None
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} doesn't exist.")

    with open(file_path, 'r') as f:
        config = json.load(f)
    return config
