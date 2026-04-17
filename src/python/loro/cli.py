
import json
import click
from .midi import MidiParser
from .dataset import EventDataset
from .model import Normalizer, RMDN
from .pipeline import Pipeline
from .utils import (validate_path,
                    echo,
                    DEVICE,
                    COLORS)
from .session import Session


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
              help='Length of sequence segments to use during training.',
              type=click.Path(file_okay=True, dir_okay=False))
@click.option('--context',
              default=200,
              help='Length of sequence segments to use during training.')
@click.option('--split',
              default=0.7,
              help='Training split factor.')
@click.option('--mixtures',
              default=10,
              help='Number of Gaussian mixtures in the model.')
@click.option('--hidden-size',
              default=256,
              help='Hidden size.')
@click.option('--batch-size',
              default=32,
              help='Batch size.')
@click.option('--lr',
              default=0.0025,
              help='Learning rate.')
@click.option('--patience',
              default=15,
              help='Number of iterations the model is allowed to not improve before stopping training.')
@click.option('--epochs',
              default=1000,
              help='Maximum number of epochs.')
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
        config_path = validate_path(input, '.json')
        with open(config_path, 'r') as f:
            config: dict = json.load(f)
        if 'input' not in config:
            raise RuntimeError(click.style(
                "config .json file must provide an input path to MIDI data.", fg=COLORS['error']))
        midi_file = config.pop('input')
    else:
        midi_file = input
        config = {}
    params = {**kwargs, **config}
    parser = MidiParser(midi_file)
    data = parser.events().to(DEVICE)
    scaler = Normalizer(data)
    dataset = EventDataset(data=data,
                           context_length=params['context'],
                           split=params['split'])
    model = RMDN(k=params['mixtures'],
                 input_size=dataset.dims,
                 dropout=params['dropout'],
                 slope=params['slope'],
                 device=DEVICE)
    pipeline = Pipeline(
        model=model,
        scaler=scaler,
        dataset=dataset,
        batch_size=params['batch_size'],
        lr=params['lr'],
        betas=tuple(params['betas']),
    )
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
        echo("\nClosing session...")
        return


main.add_command(train)
main.add_command(run)
