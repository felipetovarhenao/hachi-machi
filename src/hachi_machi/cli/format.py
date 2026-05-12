import click
from .middleware import ClickMiddleware as M
from ..midi import MidiParser
from ..io import FileIO


@click.command()
@click.argument('input',
                type=click.Path(file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.argument('output',
                default='data.csv',
                type=click.Path(file_okay=True,
                                dir_okay=False,
                                resolve_path=False,))
@M([('input', '.mid', '.midi'),
    ('output', *FileIO.EXT)]).wrapper
def format(**params):
    """Convert a MIDI file into a CSV, JSON, or TXT data format, to be used as training data.
    """
    input_file = params['input']
    output_file = params['output']
    midi = MidiParser(input_file)
    tensor = midi.events()
    features = {
        "2": {
            "masked": True
        },
        "3": {
            "categorical": True
        },
    }
    FileIO.write(tensor, output_file, True, features=features)
