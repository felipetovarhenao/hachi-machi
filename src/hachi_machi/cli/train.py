import click
import torch
import json
from .. import nn
from ..data import EventDataset
from ..nn import transforms as T
from ..trainer import Trainer
from ..features import FeatureMap
from .middleware import ClickMiddleware as M
from ..operations import DataAugmenter
from ..io import FileIO


@click.command(context_settings={'show_default': True})
@click.argument('input',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.argument('output',
                default='model.pt',
                type=click.Path(file_okay=True, dir_okay=False))
@click.option('--mixtures',
              default=10,
              type=int,
              help='Number of Gaussian mixtures in the model.')
@click.option('--layers',
              default=1,
              help='Number of recurrent layers.',
              type=int)
@click.option('--hidden-size',
              default=120,
              type=int,
              help='Number of dimensions to use for hidden representation.')
@click.option('--context',
              default=200,
              type=int,
              help='Length of sequence segments to use during training.')
@click.option('--epochs',
              default=1000,
              help='Maximum number of epochs.')
@click.option('--batch-size',
              default=32,
              help='Batch size.')
@click.option('--lr',
              default=0.0025,
              help='Learning rate.')
@click.option('--patience',
              default=15,
              help='Number of iterations the model is allowed to not improve before stopping training.')
@click.option('--dropout',
              default=0.25,
              help='Dropout rate.')
@click.option('--betas',
              default=[0.9, 0.99],
              help='Betas for AdamW (Adaptive Moment Estimation) optimizer.',
              type=click.FloatRange(0.1, 0.995),
              nargs=2)
@click.option('--slope',
              default=1e-5,
              type=click.FloatRange(0, max_open=True),
              help='Negative slope for Leaky ReLU activations.')
@click.option('--seed',
              default=1,
              help='Random seed. Use 0 for non-deterministic training.')
@click.option('--transforms', '-t',
              type=click.Choice(T.TransformFactory.options()),
              help='Optional transform layers.',
              multiple=True)
@click.option('--operations', '-op',
              type=str,
              help='Data augmentation operation(s) to stochastically apply during training.',
              multiple=True)
@M([
    ('input', *FileIO.EXT),
    ('output', '.pt')
]).wrapper
def train(**params):
    """Given a path to an **INPUT** sequential dataset, generates an pre-trained `.pt` **OUTPUT** model, trained on that dataset. The dataset must be provided as a JSON file, with the following schema:

    ```json title="data.json" showLineNumbers
    {
        "data": [
            [<feature_0>, <feature_1>, ..., <feature_N>],
            ...
            [<feature_0>, <feature_1>, ..., <feature_N>],
        ]
    }
    ```

    If the dataset is temporal, you must provide timestamps for each step in the sequence, in seconds:

    ```json title="timed-data.json" showLineNumbers
    {
        "time": [
            0.0,
            0.25,
            1.25,
            ...
        ],
        "data": [
            ...
        ]
    }
    ```

    Optionally, you may provide information about how the features should be treated by the model:

    ```json title="data-features.json" showLineNumbers
    {
        "features": {
            "0": { "categorical": true },
            "2": { "masked": true },
        },
        "data": [
            [0, 0.25, 0.1],
            [1, 0.125, 0.2],
            [0, 0.5, 0.5],
            ...
        ]
    }
    ```

    In this example, we're telling the model that feature at index `0` is categorical, and that feature at index `2` should be masked. 
    This means that this feature is output only, and the model will learn how to predict it.

    """
    device = params['device']
    seed = params['seed']
    file_path: str = params['input']
    operations = params['operations']

    if seed != 0:
        torch.manual_seed(params['seed'])

    augmenter = None

    data, feature_map = FileIO.read(file_path, device)

    if len(operations) > 0:
        augmenter = DataAugmenter(operations=operations,
                                  feature_map=feature_map)

    factory = T.TransformFactory(feature_map=feature_map)
    input_layer, output_layer = factory.make(data=data,
                                             transforms=params['transforms'])
    dataset = EventDataset(data=data,
                           input_dims=feature_map.input_dims(),
                           output_dims=feature_map.output_dims(),
                           context_length=params['context'],
                           augmenter=augmenter)
    rnn = nn.RecurrentMDN(k=params['mixtures'],
                          input_size=input_layer.output_size,
                          output_size=output_layer.output_size,
                          num_layers=params['layers'],
                          hidden_size=params['hidden_size'],
                          dropout=params['dropout'],
                          slope=params['slope'],
                          device=device)
    model = nn.PerformerModel(rnn=rnn,
                              input_layer=input_layer,
                              output_layer=output_layer,
                              input_mask=feature_map.input_dims(),
                              temporal=feature_map.temporal(),
                              device=device,)
    trainer = Trainer(model=model,
                      dataset=dataset,
                      batch_size=params['batch_size'],
                      lr=params['lr'],
                      betas=tuple(params['betas']),)
    trainer.run(file=params['output'],
                epochs=params['epochs'],
                patience=params['patience'])
