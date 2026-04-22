import torch
import click
from ..utils import tensor_to_txt, device_option, clean_params
from ..midi import MidiParser
from ..augment import MidiAugmentator
from ..console import Console


@click.command()
@click.argument('input')
@click.argument('output')
@click.option('--transform', '-t',
              default=[],
              type=click.Choice(MidiAugmentator.options()),
              multiple=True)
@click.option('--seed', '-s', default=0)
@device_option()
def render(**kwargs):
    params = clean_params(
        params=kwargs,
        file_keys=[
            ('input', ['.mid', '.midi']),
            ('output', ['.mid', '.midi', '.txt']),
        ])

    if len(params['transform']) == 0:
        params['transform'] = None

    device = params['device']
    input = kwargs['input']
    output = kwargs['output']
    seed = kwargs['seed']
    transforms = kwargs['transform']

    if seed != 0:
        torch.manual_seed(seed)
    midi = MidiParser(file=input)
    events = midi.events().to(device)
    aug = MidiAugmentator(num_voices=midi.numvoices(),
                          transforms=transforms)

    if transforms is not None:
        for name in transforms:
            name = f"use_{name.replace('-', '_')}"
            cb = getattr(aug, name)
            events = cb(events.clone())
    if not output.endswith('.txt'):
        MidiParser.render(events, output)
    else:
        tensor_to_txt(events, output)
    Console.success("DONE", bold=True)
