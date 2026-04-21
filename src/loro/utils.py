import importlib.metadata
import os
import torch
import json
from typing import Callable
from .console import Console

VERSION = importlib.metadata.version("loro")

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def validate_path(file, ext: str | list) -> bool:
    if type(ext) == str:
        ext = [ext]

    file = os.path.abspath(os.path.expanduser(file))
    file_ext = os.path.splitext(file)[1]
    if file_ext not in ext:
        raise TypeError(
            f"Invalid extension: {file_ext}. Expected: {', '.join(ext)}")
    return file


def load_config(file: str):
    file = validate_path(file, '.json')
    os.chdir(os.path.dirname(file))
    with open(file, 'r') as f:
        config: dict = json.load(f)
    if 'input' not in config:
        raise RuntimeError(
            "config .json file must provide an input path to MIDI data.")
    return config


def safe_handler(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            Console.error(e)
    return wrapper
