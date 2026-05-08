import torch
import click
from ..utils import tensor_to_txt
from ..midi import MidiParser
from ..console import Console
from .middleware import ClickMiddleware as M


@click.command(context_settings={'show_default': True})
@click.argument('input', type=click.Path(exists=True,
                                         file_okay=True,
                                         dir_okay=False,
                                         resolve_path=True,)
                )
@click.argument('output',
                default='output.mid',
                type=click.Path(file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.option('--seed', '-s', default=0)
@M([
    ('input', '.mid', '.midi'),
    ('output', '.mid', '.midi', '.txt'),
]).wrapper
def render(**params):
    """Renders a MIDI file, with optional data augmentation.

    Arguments:

    INPUT: Path to MIDI file

    OUTPUT: Path to output MIDI file
    """

    if len(params['augmentation']) == 0:
        params['augmentation'] = None

    device = params['device']
    input = params['input']
    output = params['output']
    seed = params['seed']

    if seed != 0:
        torch.manual_seed(seed)
    midi = MidiParser(file=input)
    events = midi.events().to(device)

    if not output.endswith('.txt'):
        MidiParser.render(events, output)
    else:
        tensor_to_txt(events, output)
    Console.success("DONE", bold=True)
