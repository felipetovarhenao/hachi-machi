import os
import torch
import json
import click
import traceback
from typing import Callable, Any
from .console import Console


def get_available_devices() -> list[str]:
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])

    if torch.backends.mps.is_available():
        devices.append("mps")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.extend([f"xpu:{i}" for i in range(torch.xpu.device_count())])

    devices.insert(0, "auto")

    return devices


DEVICE_OPTIONS = get_available_devices()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def safe_handler(func: Callable[[str, Any], None]) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            Console.error(traceback.format_exc())
    wrapper.__name__ = func.__name__.replace("handle_", "")
    return wrapper


def tensor_to_txt(x: torch.Tensor, output: str) -> None:
    output = validate_path(output, '.txt')
    out = ""
    col_len = x.max(0).values.log10().clip(1).ceil().int() + 2
    col_len = col_len.tolist()

    def col(x, i):
        x = str(x)
        x += " " * max(1, col_len[i] - len(x))
        return x
    for e in x.int().tolist():
        out += f'{"".join(col(x, i) for i, x in enumerate(e))}\n'
    with open(output, 'w') as f:
        f.write(out)


def device_option():
    return click.option('--device', '-d',
                        type=click.Choice(DEVICE_OPTIONS),
                        default=resolve_device('auto'),
                        help='Compute device')


def clean_params(params: dict,
                 file_keys: list[str, str] | None = None):
    if 'device' in params:
        params['device'] = resolve_device(params['device'])
    if file_keys is not None:
        for (key, ext) in file_keys:
            if key not in params:
                continue
            params[key] = validate_path(file=params[key],
                                        ext=ext)
    Console.pretty(params, header='Settings')
    return params


def progress(n: int, N: int = 10, size: int = 12):
    t = min(1, n / N)
    p = round(t * 100)
    i = int(round(size * t))
    l = "⣿" * i
    r = "⣀" * (size - i)
    return f'{l}{r} {p:.1f}%'
