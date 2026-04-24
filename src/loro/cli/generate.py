import click
import torch
from ..midi import MidiParser
from ..nn import MultiplayerAgent
from ..console import Console
from ..utils import (device_option,
                     tensor_to_txt,
                     clean_params)


@click.command()
@click.argument('model', type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument('output', default='out.txt', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--tokens',  default=100, help='Number of tokens to generate.')
@click.option('--seed', default=0, help='Random seed.')
@click.option('--temp', default=1, help='Temperature.')
@device_option()
def generate(**kwargs):
    """MODEL: Path to PyTorch model

    OUTPUT: Path to output file, in either MIDI or TXT format
    """
    params = clean_params(
        params=kwargs, file_keys=[
            ('model', '.pt'),
            ('output', ('.mid', '.midi', '.txt'))
        ])
    device = params['device']
    model_path = params['model']
    output = params['output']
    is_txt = output.endswith('.txt')
    if not is_txt:
        return
    seed = kwargs['seed']
    if seed != 0:
        torch.manual_seed(kwargs['seed'])

    agent: MultiplayerAgent = torch.load(f=model_path,
                                         weights_only=False,
                                         map_location=device)
    model = agent.model
    x_scaler, y_scaler = agent.x_scaler, agent.y_scaler
    model.eval()
    hidden = None

    events = []
    with torch.no_grad():
        x = torch.randn(1, 1, model.input_size, device=device)
        x = x_scaler(x, inverse=True)
        for _ in range(kwargs['tokens']):
            y, hidden = model.step(x=x_scaler(x),
                                   hidden=hidden,
                                   temp=kwargs['temp'])
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
