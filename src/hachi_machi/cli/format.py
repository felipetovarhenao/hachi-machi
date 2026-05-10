import click
from .middleware import ClickMiddleware as M
from ..midi import MidiParser
import json


@click.command()
@click.argument('input',
                type=click.Path(file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.argument('output',
                default='data.json',
                type=click.Path(file_okay=True,
                                dir_okay=False,
                                resolve_path=False,))
@M([('input', '.mid', '.midi'), ('output', '.json')]).wrapper
def format(**params):
    """Convert a MIDI file into a JSON dataset, to be used as training data.

    INPUT: MIDI file to format as JSON dataset

    OUTPUT: Path to output JSON file
    """
    input_file = params['input']
    output_file = params['output']
    midi = MidiParser(input_file)
    data = midi.events()
    data[..., 0] = data[..., 0].cumsum(dim=0)
    time = data[..., 0]
    data = data[..., 1:]
    content = {
        "features": {
            "2": {
                "masked": True
            },
            "3": {
                "categorical": True
            },
        },
        "time": time.tolist(),
        "data": data.tolist()
    }

    with open(output_file, 'w') as f:
        json.dump(obj=content, fp=f, indent=4)
