import os
import torch
import traceback
from typing import Callable, Any
from .console import Console


def validate_path(file, ext: str | list) -> str:
    if type(ext) == str:
        ext = [ext]
    file = os.path.abspath(os.path.expanduser(file))
    file_ext = os.path.splitext(file)[1]
    if file_ext not in ext:
        raise TypeError(
            f"Invalid extension: {file_ext}. Expected: {', '.join(ext)}")
    return file


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


def progress(n: int, N: int = 10, size: int = 12):
    t = min(1, n / N)
    p = round(t * 100)
    i = int(round(size * t))
    l = "⣿" * i
    r = "⣀" * (size - i)
    return f'{l}{r} {p:.1f}%'
