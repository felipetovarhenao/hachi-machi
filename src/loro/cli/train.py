import click
import torch
from ..augment import MidiAugmentator
from ..midi import MidiParser
from ..data import EventDataset
from ..nn import FeatureScaler, RecurrentMDN
from ..pipeline import Pipeline
from ..console import Console
from ..utils import (load_config,
                     device_option,
                     clean_params)


@click.command()
@click.argument('input',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.option('--output', '-o',
              default='model.pt',
              help='Output path for trained Pytorch model (.pt).',
              type=click.Path(file_okay=True, dir_okay=False))
@click.option('--mixtures',
              default=10,
              help='Number of Gaussian mixtures in the model.')
@click.option('--hidden-size',
              default=256,
              help='Hidden size.')
@click.option('--context',
              default=200,
              help='Length of sequence segments to use during training.')
@click.option('--split',
              default=0.7,
              help='Training split factor.')
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
              help='Negative slope for Leaky ReLU activations.')
@click.option('--seed',
              default=1,
              help='Random seed.')
@click.option('--transform', '-t',
              default=MidiAugmentator.options(),
              type=click.Choice(MidiAugmentator.options()),
              help='Data augmentation transform to randomly apply during training.',
              multiple=True)
@device_option()
def train(input, **kwargs):
    if input.endswith('.json'):
        config = load_config(input)
        midi_file = config['input']
    else:
        midi_file = input
        config = {}
    params = clean_params(
        params={**kwargs, **config},
        file_keys=[
            ('input', ['.json', '.mid', '.midi']),
            ('output', '.pt')
        ]
    )
    device = params['device']
    seed = params['seed']
    if seed != 0:
        torch.manual_seed(params['seed'])

    Console.action("\nParsing MIDI...", italic=True)

    parser = MidiParser(midi_file)
    if parser.numvoices() < 2:
        raise RuntimeError(
            "MIDI file must contain of two or more channels, one for each player.")
    data = parser.events().to(device)
    x_scaler = FeatureScaler(data[..., :-1], time_dims=[0, 1])
    y_scaler = FeatureScaler(data, time_dims=[0, 1, -1])
    augmentator = MidiAugmentator(num_voices=parser.numvoices(),
                                  transforms=params['transform'])
    dataset = EventDataset(data=data,
                           context_length=params['context'],
                           split=params['split'],
                           augmentator=augmentator)
    model = RecurrentMDN(k=params['mixtures'],
                         input_size=dataset.input_size,
                         output_size=dataset.output_size,
                         dropout=params['dropout'],
                         slope=params['slope'],
                         device=device)
    pipeline = Pipeline(model=model,
                        x_scaler=x_scaler,
                        y_scaler=y_scaler,
                        dataset=dataset,
                        batch_size=params['batch_size'],
                        lr=params['lr'],
                        betas=tuple(params['betas']),)
    Console.action(
        f"{parser.numvoices()} players found", italic=True)
    pipeline.run(file=params['output'],
                 epochs=params['epochs'],
                 patience=params['patience'])
