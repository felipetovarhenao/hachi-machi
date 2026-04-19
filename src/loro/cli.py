import click
from .augment import MidiAugmentator
from .midi import MidiParser
from .dataset import EventDataset
from .model import FeatureScaler, RecurrentMDN
from .pipeline import Pipeline
from .session import Session
from .console import Console
from .utils import (validate_path,
                    load_config,
                    DEVICE,)


@click.group()
def main():
    pass


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
              help='Betas for AdamW (Adaptive Moment Estimation) optimizer.', multiple=True)
@click.option('--slope',
              default=0.001,
              help='Negative slope for Leaky ReLU activations.')
def train(input, **kwargs):
    global DEVICE
    if input.endswith('.json'):
        config = load_config(input)
        midi_file = config.pop('input')
    else:
        midi_file = input
        config = {}
    params = {**kwargs, **config}
    parser = MidiParser(midi_file)
    if parser.numvoices() < 2:
        raise RuntimeError(
            "MIDI file must contain of two or more channels, one for each player.")
    data = parser.events().to(DEVICE)
    scaler = FeatureScaler(data)
    augmentator = MidiAugmentator(num_voices=parser.numvoices())
    dataset = EventDataset(data=data,
                           context_length=params['context'],
                           split=params['split'],
                           augmentator=augmentator)
    model = RecurrentMDN(k=params['mixtures'],
                         input_size=dataset.dims,
                         dropout=params['dropout'],
                         slope=params['slope'],
                         device=DEVICE)
    pipeline = Pipeline(model=model,
                        scaler=scaler,
                        dataset=dataset,
                        batch_size=params['batch_size'],
                        lr=params['lr'],
                        betas=tuple(params['betas']),)
    pipeline.run(file=params['output'],
                 epochs=params['epochs'],
                 patience=params['patience'])


@click.command()
@click.argument('input')
@click.option('--in-port', default=8000, help='Input OSC port.')
@click.option('--out-port', default=9000, help='Output OSC port.')
@click.option('--address', default='127.0.0.1', help='OSC address')
def run(input, **kwargs):
    global DEVICE
    config = kwargs
    model = validate_path(input, '.pt')
    session = Session(model=model,
                      in_port=config['in_port'],
                      out_port=config['out_port'],
                      host=config['address'],
                      device=DEVICE)
    try:
        session.start()
    except KeyboardInterrupt:
        Console.action("\nClosing session...")
        return


main.add_command(train)
main.add_command(run)
