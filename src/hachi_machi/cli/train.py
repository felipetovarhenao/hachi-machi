import click
import torch
import json
from .. import nn
from ..midi import MidiParser
from ..data import EventDataset
from ..nn import transforms as T
from ..trainer import Trainer
from ..features import FeatureMap
from .middleware import ClickMiddleware as M
from ..augment import DataAugmentator


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
    ('input', '.json'),
    ('output', '.pt')
]).wrapper
def train(**params):
    """Trains a model on custom sequential data. 
The data must be provided as a JSON file, and each event in the sequence must structured as follows:

[<ms_time> <voice_id> <feature_1> ... <feature_N> ]

Note that <voice_id> must be an zero-based integer.

Arguments:

    INPUT: Path to JSON file to use as training data

    OUTPUT: Output path for trained Pytorch model (.pt)
    """
    device = params['device']
    seed = params['seed']
    file_path: str = params['input']
    augmentator = None
    if seed != 0:
        torch.manual_seed(params['seed'])
    temporal = True
    with open(file_path, 'r') as f:
        content = json.load(f)

    if not isinstance(content, dict) or 'data' not in content:
        raise TypeError(
            f'Invalid data. Format sequence under "data" key and provide sequence as a 2D matrix.')

    data = content['data']
    features: dict = content.get('features', dict())

    try:
        data = torch.tensor(data, dtype=torch.float32).to(device)
    except:
        raise ValueError(
            "data must be structured as a 2D matrix, each row with the same number of elements")

    if 'time' not in content:
        temporal = False
    else:
        time = torch.tensor(
            content['time'], dtype=torch.float32).reshape(-1, 1).to(device)
        data = torch.cat([time, data], dim=-1)
        data[1:, 0] = data[..., 0].diff(dim=-1)
        features = {str(int(k) + 1): v for (k, v) in features.items()}
        features = {
            '0': {
                'type': 'temporal'
            },
            **features
        }

    feature_map = FeatureMap(data, features)
    operations = params['operations']

    if len(operations) > 0:
        augmentator = DataAugmentator(operations=operations,
                                      feature_map=feature_map)

    factory = T.TransformFactory(feature_map=feature_map)
    input_layer, output_layer = factory.make(data=data,
                                             transforms=params['transforms'])
    dataset = EventDataset(data=data,
                           input_dims=feature_map.input_dims(),
                           output_dims=feature_map.output_dims(),
                           context_length=params['context'],
                           augmentator=augmentator)
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
                              temporal=temporal,
                              device=device,)
    trainer = Trainer(model=model,
                      dataset=dataset,
                      batch_size=params['batch_size'],
                      lr=params['lr'],
                      betas=tuple(params['betas']),)
    trainer.run(file=params['output'],
                epochs=params['epochs'],
                patience=params['patience'])
