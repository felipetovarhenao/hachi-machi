import toml
import os
import torch
from ..utils import validate_path
from ..console import Console
from ..nn import transforms as T
import click
import functools


class ClickMiddleware:

    def __init__(self,
                 path_args: list | None = None):
        self.path_args = path_args

    def wrapper(self, func):
        func.__doc__ = Console.info(func.__doc__, defer=True)

        @click.option('--device', '-d',
                      type=click.Choice(self.get_available_devices()),
                      default='auto',
                      help='Compute device')
        @functools.wraps(func)
        def _wrapper(**kwargs):
            config = self._parse(**kwargs)
            Console.pretty(config, header='Settings')
            func(**config)

        return _wrapper

    def _parse(self, **params) -> dict:
        if self.path_args is not None:
            for (key, *ext) in self.path_args:
                if key not in params:
                    continue
                param = params[key]
                if param is None and None in ext:
                    continue
                if isinstance(param, str) and param.endswith('.toml'):
                    config = {**params, **self.from_file(param)}
                    return self._parse(**config)
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
        file = validate_path(file, '.toml')
        os.chdir(os.path.dirname(file))
        with open(file, 'r') as f:
            config: dict = toml.load(f)
        return config

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

    def train_wrapper(self, func):
        @click.argument('input',
                        type=click.Path(exists=True,
                                        file_okay=True,
                                        dir_okay=False,
                                        resolve_path=True,))
        @click.argument('output',
                        default='model.pt',
                        type=click.Path(file_okay=True, dir_okay=False))
        @click.option('--mixtures',
                      default=10,
                      type=int,
                      help='Number of Gaussian mixtures in the model.')
        @click.option('--layers',
                      default=1,
                      help='Number of recurrent layers.',
                      type=int)
        @click.option('--hidden-size',
                      default=120,
                      type=int,
                      help='Number of dimensions to use for hidden representation.')
        @click.option('--context',
                      default=200,
                      type=int,
                      help='Length of sequence segments to use during training.')
        @click.option('--split',
                      default=0.7,
                      type=float,
                      help='Training split factor.')
        @click.option('--epochs',
                      default=1000,
                      help='Maximum number of epochs.')
        @click.option('--batch-size',
                      default=32,
                      help='Batch size.')
        @click.option('--lr',
                      default=0.0025,
                      help='Learning rate.')
        @click.option('--patience',
                      default=15,
                      help='Number of iterations the model is allowed to not improve before stopping training.')
        @click.option('--dropout',
                      default=0.25,
                      help='Dropout rate.')
        @click.option('--betas',
                      default=[0.9, 0.99],
                      help='Betas for AdamW (Adaptive Moment Estimation) optimizer.',
                      type=click.FloatRange(0.1, 0.995),
                      nargs=2)
        @click.option('--slope',
                      default=1e-5,
                      type=click.FloatRange(0, max_open=True),
                      help='Negative slope for Leaky ReLU activations.')
        @click.option('--seed',
                      default=1,
                      help='Random seed. Use 0 for non-deterministic training.')
        @click.option('--features', '-f',
                      default=['categorical', 'normalize'],
                      type=click.Choice(T.TransformFactory.options()),
                      help='Feature transform layers.',
                      multiple=True)
        @self.wrapper
        @functools.wraps(func)
        def _wrapper(**kwargs):
            func(**kwargs)
        return _wrapper
