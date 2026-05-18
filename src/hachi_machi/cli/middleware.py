import toml
import os
import torch
from ..utils import validate_path
from ..console import Console
import click
import functools
import re
import yaml


class ClickMiddleware:

    def __init__(self,
                 path_args: list | None = None,
                 device: int = 'auto'):
        self.path_args = path_args
        self.device = device

    def wrapper(self, func):
        @click.option('--device', '-d',
                      type=click.Choice(self.get_available_devices()),
                      default=self.device,
                      help='Compute device')
        @click.option('--debug',
                      flag_value=True,
                      default=False,
                      help="Debug mode. Includes traceback when an error is raised.")
        @functools.wraps(func)
        def _wrapper(**kwargs):
            config = self._parse(**kwargs)
            debug = config.pop('debug')
            Console.pretty(config, header='Settings')
            if not debug:
                try:
                    func(**config)
                except Exception as e:
                    Console.error(e.args[0])
            else:
                func(**config)

        return _wrapper

    def _parse(self, **params) -> dict:
        if self.path_args is not None:
            for (key, *ext) in self.path_args:
                if key not in params:
                    continue
                params[key] = validate_path(file=params[key],
                                            ext=ext)
        if 'device' in params:
            params['device'] = self.resolve_device(params['device'])
        return params

    @staticmethod
    def resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def from_file(file: str):
        file = validate_path(file, ('.toml', '.yaml', '.yml'))
        os.chdir(os.path.dirname(file))
        load = toml.load if file.endswith('.toml') else yaml.safe_load
        with open(file, 'r') as f:
            data: dict = load(f)
        cfg = {}
        for k, v in data.items():
            k = re.sub(r"[ -_]", "_", k)
            cfg[k] = v
        return cfg

    @staticmethod
    def get_available_devices() -> list[str]:
        devices = ["cpu"]

        if torch.cuda.is_available():
            devices.extend(
                [f"cuda:{i}" for i in range(torch.cuda.device_count())])

        if torch.backends.mps.is_available():
            devices.append("mps")

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices.extend(
                [f"xpu:{i}" for i in range(torch.xpu.device_count())])

        devices.insert(0, "auto")

        return devices
