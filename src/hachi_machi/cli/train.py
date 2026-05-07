import click
import torch
import json
from .. import nn
from ..augment import MidiAugmentator
from ..midi import MidiParser
from ..data import EventDataset
from ..nn import transforms as T
from ..trainer import Trainer
from ..features import FeatureMap
from .middleware import ClickMiddleware as M


@click.command(context_settings={'show_default': True})
@click.option('--augmentation', '-a',
              type=click.Choice(MidiAugmentator.options()),
              help='Data augmentation techniques to stochastically apply during training.',
              multiple=True)
@M([
    ('input', '.mid', '.midi'),
    ('output', '.pt')
]).train_wrapper
def train(**params):
    """Trains a model on MIDI data.

    Arguments:

    INPUT: Path to MIDI file to use as training data

    OUTPUT: Output path for trained Pytorch model (.pt)
    """
    _train(**params)


@click.command(name='train-custom',
               context_settings={'show_default': True})
@M([
    ('input', '.json'),
    ('output', '.pt')
]).train_wrapper
def train_custom(**params):
    """Trains a model on custom sequential data. 
The data must be provided as a JSON file, and each event in the sequence must structured as follows:

[<ms_time> <voice_id> <feature_1> ... <feature_N> ]

Note that <voice_id> must be an zero-based integer.

Arguments:

    INPUT: Path to JSON file to use as training data

    OUTPUT: Output path for trained Pytorch model (.pt)
    """
    _train(**params)


def _train(**params):
    device = params['device']
    seed = params['seed']
    file_path: str = params['input']
    augmentator = None
    if seed != 0:
        torch.manual_seed(params['seed'])
    temporal = True
    if file_path.endswith('.json'):
        with open(params['input'], 'r') as f:
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
    else:
        parser = MidiParser(file_path)
        num_channels = len(parser.channels)
        if num_channels < 2:
            raise RuntimeError(
                "MIDI file must contain of two or more channels.")
        data = parser.events().to(device)
        feature_map = FeatureMap(data, features={
            "0": {
                "type": "categorical"
            },
            "3": {
                'masked': True,
                'type': 'temporal'
            }
        })
        augmentation = params['augmentation']
        if len(augmentation) > 0:
            augmentator = MidiAugmentator(channels=parser.channels,
                                          augmentation=augmentation)

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
