import click
import torch
from ..midi import MidiParser
from ..nn import MultiplayerAgent
from ..console import Console
from ..utils import (tensor_to_txt,
                     progress,)
from .config import Config


@click.command()
@click.argument('model', type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument('output', default='out.txt', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--tokens',  default=100, help='Number of tokens to generate.')
@click.option('--seed', default=0, help='Random seed.')
@click.option('--temp',
              default=1,
              type=click.FloatRange(0.02, max_open=True),
              help='Temperature')
@Config([
    ('model', '.pt'),
    ('output', '.mid', '.midi', '.txt')
]).parse
def generate(**params):
    """MODEL: Path to PyTorch model

    OUTPUT: Path to output file, in either MIDI or TXT format
    """
    device = params['device']
    model_path = params['model']
    output = params['output']
    is_txt = output.endswith('.txt')
    if not is_txt:
        return
    seed = params['seed']
    if seed != 0:
        torch.manual_seed(params['seed'])

    agent: MultiplayerAgent = torch.load(f=model_path,
                                         weights_only=False,
                                         map_location=device)
    model = agent.model
    x_scaler, y_scaler = agent.x_scaler, agent.y_scaler
    model.eval()
    hidden = None

    events = []
    num_tokens = params['tokens']
    display = Console.get_display(n_rows=1)
    with torch.no_grad():
        x = torch.randn(1, 1, model.input_size, device=device)
        x = x_scaler(x, inverse=True)
        for i in range(num_tokens):
            display.update(progress=progress(i, num_tokens - 1))
            y, hidden = model.step(x=x_scaler(x),
                                   hidden=hidden,
                                   temp=params['temp'])
            x: torch.Tensor = y_scaler(y.clone(), inverse=True)
            x = x.clip(0).round()
            events.append(x.squeeze())
            x = x[..., :-1]

    events = torch.stack(events).float()

    if is_txt:
        tensor_to_txt(events, output)
    else:
        MidiParser.render(events=events,
                          output_path=output)

    Console.success("\nDONE", bold=True)
