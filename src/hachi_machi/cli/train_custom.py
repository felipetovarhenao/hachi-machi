import click
import torch
import json
from .config import Config
from ..data import EventDataset
from ..trainer import Trainer
from .. import nn
from ..nn import transforms as T


@click.command(name='train-custom')
@Config([
    ('input', '.json'),
    ('output', '.pt')
]).train_options
def train_custom(**params):
    device = params['device']
    with open(params['input'], 'r') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"Invalid data format: {type(data)}")
    try:
        data = torch.tensor(data).to(device)
    except:
        raise ValueError(
            "data must be structured as a 2D matrix, each row with the same number of elements")
    VOICE_DIM, TIME_DIM = 1, 0
    data[1:, TIME_DIM] = data[..., TIME_DIM].diff(dim=-1)
    voices = data[..., VOICE_DIM].unique()
    num_voices = len(voices)
    dataset = EventDataset(data=data,
                           context_length=params['context'],
                           split=params['split'],)

    factory = T.TransformFactory(voice_dim=VOICE_DIM,
                                 time_dim=TIME_DIM,
                                 num_voices=num_voices)
    input_layer, output_layer = factory.make(input_data=data[..., :-1].clone(),
                                             output_data=data.clone(),
                                             transforms=['normalize'])

    rnn = nn.RecurrentMDN(k=params['mixtures'],
                          input_size=input_layer.output_size,
                          output_size=output_layer.output_size,
                          num_layers=params['layers'],
                          hidden_size=params['hidden_size'],
                          dropout=params['dropout'],
                          slope=params['slope'],
                          device=device)
    model = nn.MultiplayerAgent(model=rnn,
                                input_layer=input_layer,
                                output_layer=output_layer,
                                num_voices=num_voices,
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
