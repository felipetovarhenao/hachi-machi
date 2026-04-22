import click
import torch
from ..midi import MidiParser
from ..model import MusicAgent
from ..console import Console
from ..utils import (validate_path,
                     tensor_to_txt,
                     DEVICE)


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument('output', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--tokens',  default=100, help='Number of tokens to generate.')
@click.option('--seed', default=0, help='Random seed.')
@click.option('--temp', default=1, help='Temperature.')
def generate(model_path, output, **kwargs):
    global DEVICE
    model_path = validate_path(model_path, '.pt')
    output = validate_path(output, ['.mid', '.midi', '.txt'])
    is_txt = output.endswith('.txt')
    if not is_txt:
        return
    seed = kwargs['seed']
    if seed != 0:
        torch.manual_seed(kwargs['seed'])
    Console.pretty({'model': model_path,
                    'output': output,
                    **kwargs}, header="Settings")
    agent: MusicAgent = torch.load(f=model_path,
                                   weights_only=False,
                                   map_location=DEVICE)
    model = agent.model
    scaler = agent.scaler
    model.eval()
    hidden = None

    events = []
    with torch.no_grad():
        x = torch.randn(1, 1, model.input_size, device=DEVICE)
        x = scaler(x, inverse=True)
        for _ in range(kwargs['tokens']):
            y, hidden = model.step(x=scaler(x),
                                   hidden=hidden,
                                   temp=kwargs['temp'])
            x: torch.Tensor = scaler(y.clone(), inverse=True)
            x = x.clip(0).round()
            events.append(x.squeeze())

    events = torch.stack(events).float()

    if is_txt:
        tensor_to_txt(events, output)
    else:
        MidiParser.render(events=events,
                          output_path=output)

    Console.success("\nDONE", bold=True)
