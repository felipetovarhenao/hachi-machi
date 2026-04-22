import torch
import click
from ..utils import validate_path, tensor_to_txt
from ..midi import MidiParser
from ..augment import MidiAugmentator
from ..console import Console


@click.command()
@click.argument('input')
@click.argument('output')
@click.option('--augment', '-a', default=None, multiple=True)
@click.option('--seed', '-s', default=0)
def render(**kwargs):
    Console.pretty(kwargs, header="Render info")
    input = validate_path(kwargs['input'], ['.mid', '.midi'])
    output = validate_path(kwargs['output'], ['.mid', '.midi', '.txt'])
    seed = kwargs['seed']
    if seed != 0:
        torch.manual_seed(seed)
    midi = MidiParser(file=input)
    events = midi.events()
    aug = MidiAugmentator(midi.numvoices())
    if kwargs['augment'] is not None and len(kwargs['augment']) > 0:
        options = {}
        for method in dir(aug):
            if not method.startswith('use_'):
                continue
            name = method[4:]
            name = name.replace('_', '-')
            options[name] = getattr(aug, method)
        for name in kwargs['augment']:
            if name not in options:
                opt_list = '\n'.join(
                    [f'-a {opt}' for opt in sorted(options.keys())])
                raise ValueError(Console.error(msg=f"\nInvalid augmentation: {name}.\nUse any combination of the following:\n{opt_list}",
                                               defer=True,
                                               bold=True))
            name = f"use_{name.replace('-', '_')}"
            cb = getattr(aug, name)
            events = cb(events.clone())
        is_txt = output.endswith('.txt')
    if not is_txt:
        MidiParser.render(events, output)
    else:
        tensor_to_txt(events, output)
    Console.success("DONE", bold=True)
