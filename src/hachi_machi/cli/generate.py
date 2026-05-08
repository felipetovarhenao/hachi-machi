import os
import click
import torch
import json
from ..midi import MidiParser
from ..nn import PerformerModel
from ..console import Console
from ..utils import (tensor_to_txt,
                     progress,)
from .middleware import ClickMiddleware as M


@click.command(name='gen', context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument('output', default='out.txt', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--tokens', '-n',  default=100, help='Number of tokens to generate.')
@click.option('--seed', default=0, help='Random seed.')
@M([
    ('model', '.pt'),
    ('output', '.txt', '.json')
], device='cpu').wrapper
def generate(**params):
    """Generates data auto-regressively given some pre-trained model.

    Arguments:

    MODEL: Path to PyTorch model

    OUTPUT: Path to output file, in either MIDI or TXT format
    """
    device = params['device']
    model_path = params['model']
    output = params['output']
    ext = os.path.splitext(output)[1]
    seed = params['seed']
    if seed != 0:
        torch.manual_seed(params['seed'])

    model: PerformerModel = torch.load(f=model_path,
                                       weights_only=False,
                                       map_location=device)
    model.reset()
    model.eval()
    events = []
    num_tokens = params['tokens']
    display = Console.get_display(n_rows=1)
    with torch.no_grad():
        x = torch.randn(1, 1, model.input_layer.output_size, device=device)
        x = model.input_layer(x, inverse=True)
        for i in range(num_tokens):
            display.update(progress=progress(i, num_tokens - 1))
            y = model.step(x)
            events.append(y.clone())
            x = y[..., model.input_mask]

    events = torch.cat(events, dim=1).squeeze(0)
    if model.temporal:
        events[..., 0] = events[..., 0].cumsum(0)

    if ext == '.txt':
        tensor_to_txt(events, output)
    elif ext == '.json':
        with open(output, 'w') as f:
            if model.temporal:
                content = {"time": events[..., 0].tolist(),
                           "data": events[..., 1:].tolist()}
            else:
                content = {'data': events.tolist()}
            json.dump(obj=content,
                      fp=f,
                      indent=4)

    Console.success("\nDONE", bold=True)
