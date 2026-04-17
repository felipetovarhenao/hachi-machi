import click
import os
import torch
import enum


def validate_path(file, ext: str | list) -> bool:
    if type(ext) == str:
        ext = [ext]

    file = os.path.abspath(os.path.expanduser(file))
    file_ext = os.path.splitext(file)[1]
    if file_ext not in ext:
        raise TypeError(
            f"Invalid extension: {file_ext}. Expected: {', '.join(ext)}")
    return file


COLORS = {
    'success': (85, 219, 158),
    'error': (234, 79, 79),
    'info': (120, 190, 250),
    'neutral': (141, 165, 166),
}


def echo(text: str, type: int = 'neutral'):
    global COLORS
    color = COLORS[type]
    click.echo(click.style(text=text, fg=color))


DEVICE = torch.accelerator.current_accelerator()
