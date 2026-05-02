import click
import torch
from ..augment import MidiAugmentator
from ..midi import MidiParser
from ..data import EventDataset
from ..nn import RecurrentMDN, MultiplayerAgent
from ..nn import transforms as T
from ..trainer import Trainer
from ..console import Console
from ..features import FeatureMap
from .middleware import ClickMiddleware as M


@click.command(context_settings={'show_default': True})
@click.option('--transform', '-t',
              default=MidiAugmentator.options(),
              type=click.Choice(MidiAugmentator.options()),
              help='Data augmentation transform to randomly apply during training.',
              multiple=True)
@M([
    ('input', '.mid', '.midi'),
    ('output', '.pt')
]).train_wrapper
def train(**params):
    """Trains a model on MIDI data.

    INPUT: Path to MIDI file to use as training data

    OUTPUT: Output path for trained Pytorch model (.pt)
    """
    device = params['device']
    seed = params['seed']
    midi_file = params['input']
    if seed != 0:
        torch.manual_seed(params['seed'])
    parser = MidiParser(midi_file)
    num_channels = len(parser.channels)
    if num_channels < 2:
        raise RuntimeError(
            "MIDI file must contain of two or more channels.")
    data = parser.events().to(device)

    feature_map = FeatureMap(data, features={
        "2": {'masked': True}
    })

    VOICE_DIM = 1

    factory = T.TransformFactory(feature_map=feature_map)
    transforms = params['features']
    input_layer, output_layer = factory.make(data=data,
                                             transforms=transforms)

    augmentator = MidiAugmentator(channels=parser.channels,
                                  transforms=params['transform'])
    dataset = EventDataset(data=data,
                           input_dims=feature_map.input_dims(),
                           output_dims=feature_map.output_dims(),
                           context_length=params['context'],
                           split=params['split'],
                           augmentator=augmentator)
    rnn = RecurrentMDN(k=params['mixtures'],
                       input_size=input_layer.output_size,
                       output_size=output_layer.output_size,
                       num_layers=params['layers'],
                       hidden_size=params['hidden_size'],
                       dropout=params['dropout'],
                       slope=params['slope'],
                       device=device)
    model = MultiplayerAgent(rnn=rnn,
                             input_layer=input_layer,
                             output_layer=output_layer,
                             input_mask=feature_map.input_dims(),
                             voice_dim=VOICE_DIM,
                             device=device,)
    trainer = Trainer(model=model,
                      dataset=dataset,
                      batch_size=params['batch_size'],
                      lr=params['lr'],
                      betas=tuple(params['betas']),)
    trainer.run(file=params['output'],
                epochs=params['epochs'],
                patience=params['patience'])
