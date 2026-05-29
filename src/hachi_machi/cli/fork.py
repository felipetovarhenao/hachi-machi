import torch
import click
from ..console import Console
from ..ops.operator import DataOperator
from .middleware import ClickMiddleware as M
from ..io import FileIO


@click.command(context_settings={'show_default': True})
@click.argument('input', type=click.Path(exists=True,
                                         file_okay=True,
                                         dir_okay=False,
                                         resolve_path=True,))
@click.argument('output',
                default='output.csv',
                type=click.Path(file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.option('--operations', '-op',
              help='Data augmentation operation(s) to stochastically apply during training. See [operations](operations).',
              type=str,
              multiple=True)
@M.seed()
@M([
    ('input', *FileIO.EXT),
    ('output', *FileIO.EXT),
]).wrapper
def fork(**params):
    """
    Given some **INPUT** data, it generates an **OUTPUT** data file (`.txt`, `.csv`, `.json`), given any number of optional data augmentation [operations](/docs/commands/operations).
    This command is most useful as a playground to test and experiment with data augmentation pipelines before training a model, and as a shortcut for converting datasets from one file type to another.
    """

    device = params['device']
    input = params['input']
    output = params['output']
    seed = params['seed']
    ops = params['operations']

    if seed != 0:
        torch.manual_seed(seed)

    data, feature_map = FileIO.read(path=input,
                                    device=device)

    augmenter = DataOperator.from_callbacks(callbacks=ops,
                                            feature_map=feature_map)

    data = augmenter(data)

    FileIO.write(tensor=data,
                 path=output,
                 temporal=feature_map.temporal(),
                 features=feature_map.to_dict())

    Console.success("DONE", bold=True)
