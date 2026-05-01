import torch
import click
from ..utils import tensor_to_txt
from ..midi import MidiParser
from ..augment import MidiAugmentator
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
@click.option('--transform', '-t',
              default=[],
              type=click.Choice(MidiAugmentator.options()),
              multiple=True)
@click.option('--seed', '-s', default=0)
@M([
    ('input', '.mid', '.midi'),
    ('output', '.mid', '.midi', '.txt'),
]).wrapper
def render(**params):
    """INPUT: Path to MIDI file

    OUTPUT: Path to output MIDI file
    """

    if len(params['transform']) == 0:
        params['transform'] = None

    device = params['device']
    input = params['input']
    output = params['output']
    seed = params['seed']
    transforms = params['transform']

    if seed != 0:
        torch.manual_seed(seed)
    midi = MidiParser(file=input)
    events = midi.events().to(device)

    if transforms is not None:
        aug = MidiAugmentator(num_voices=midi.numvoices(),
                              transforms=transforms)
        for name in transforms:
            name = f"use_{name.replace('-', '_')}"
            cb = getattr(aug, name)
            events = cb(events.clone())
    if not output.endswith('.txt'):
        MidiParser.render(events, output)
    else:
        tensor_to_txt(events, output)
    Console.success("DONE", bold=True)
