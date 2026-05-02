import click
import torch
import json
from .middleware import ClickMiddleware as M
from ..data import EventDataset
from ..trainer import Trainer
from .. import nn
from ..nn import transforms as T
from ..features import FeatureMap


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

    INPUT: Path to JSON file to use as training data

    OUTPUT: Output path for trained Pytorch model (.pt)
    """
    device = params['device']
    with open(params['input'], 'r') as f:
        content = json.load(f)
        if isinstance(content, list):
            data = content
            features = {}
        elif isinstance(content, dict):
            if 'data' not in content:
                raise TypeError(f"Invalid data:\n{data}")
            data = content['data']
            features = content.get('features', dict())
    try:
        data = torch.tensor(data).to(device)
    except:
        raise ValueError(
            "data must be structured as a 2D matrix, each row with the same number of elements")

    TIME_DIM, VOICE_DIM = 0, 1
    data[1:, TIME_DIM] = data[..., TIME_DIM].diff(dim=-1)

    fmap = FeatureMap(data, features)
    dataset = EventDataset(data=data,
                           input_dims=fmap.input_dims(),
                           output_dims=fmap.output_dims(),
                           context_length=params['context'],
                           split=params['split'],)
    factory = T.TransformFactory(feature_map=fmap)
    input_layer, output_layer = factory.make(data=data,
                                             transforms=params['features'])

    rnn = nn.RecurrentMDN(k=params['mixtures'],
                          input_size=input_layer.output_size,
                          output_size=output_layer.output_size,
                          num_layers=params['layers'],
                          hidden_size=params['hidden_size'],
                          dropout=params['dropout'],
                          slope=params['slope'],
                          device=device)
    model = nn.MultiplayerAgent(rnn=rnn,
                                input_layer=input_layer,
                                output_layer=output_layer,
                                input_mask=fmap.input_dims(),
                                device=device,
                                voice_dim=VOICE_DIM)
    trainer = Trainer(model=model,
                      dataset=dataset,
                      batch_size=params['batch_size'],
                      lr=params['lr'],
                      betas=params['betas'])
    trainer.run(file=params['output'],
                epochs=params['epochs'],
                patience=params['patience'])
