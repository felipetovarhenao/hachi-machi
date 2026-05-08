import torch
import click
from ..utils import tensor_to_txt, load_data
from ..console import Console
from ..augment import DataAugmentator
from .middleware import ClickMiddleware as M


@click.command(context_settings={'show_default': True})
@click.argument('input', type=click.Path(exists=True,
                                         file_okay=True,
                                         dir_okay=False,
                                         resolve_path=True,))
@click.argument('output',
                default='output.txt',
                type=click.Path(file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.option('--operations', '-op',
              help='Data augmentation operation(s) to stochastically apply during training.',
              type=str,
              multiple=True)
@click.option('--seed', '-s', default=0)
@M([
    ('input', '.json'),
    ('output', '.txt'),
]).wrapper
def augment(**params):
    """Renders a MIDI file, with optional data augmentation.

    Arguments:

    INPUT: Path to JSON file

    OUTPUT: Path to output MIDI file
    """

    device = params['device']
    input = params['input']
    output = params['output']
    seed = params['seed']
    ops = params['operations']

    if seed != 0:
        torch.manual_seed(seed)

    data, feature_map, _ = load_data(file_path=input,
                                     device=device)

    augmentator = DataAugmentator(operations=ops, feature_map=feature_map)
    for op in augmentator.operations:
        data = op(data)
    tensor_to_txt(data, output)
    Console.success("DONE", bold=True)
