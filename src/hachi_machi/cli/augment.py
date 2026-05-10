import torch
import click
from ..utils import tensor_to_txt, load_data, tensor_to_csv
from ..console import Console
from ..operations import DataAugmenter
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
    ('output', '.txt', '.csv'),
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

    data, feature_map = load_data(file_path=input,
                                  device=device)

    augmenter = DataAugmenter(operations=ops, feature_map=feature_map)
    for op in augmenter.operations:
        data = op(data)
    if feature_map.temporal:
        data[..., 0] = data[..., 0].cumsum(0)
    if output.endswith('.txt'):
        tensor_to_txt(data, output)
    else:
        tensor_to_csv(data, output, feature_map.temporal)
    Console.success("DONE", bold=True)
