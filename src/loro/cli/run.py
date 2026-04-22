
import click
from ..session import Session
from ..console import Console
from ..utils import (validate_path,
                     DEVICE)


@click.command()
@click.argument('input')
@click.option('--in-port', default=8000, help='Input OSC port.')
@click.option('--out-port', default=9000, help='Output OSC port.')
@click.option('--address', default='127.0.0.1', help='OSC address')
def run(input, **kwargs):
    global DEVICE
    config = kwargs
    model = validate_path(input, '.pt')
    Console.pretty({'model': model,
                    'device': DEVICE,
                   **config},
                   header='Session info:')
    session = Session(model=model,
                      in_port=config['in_port'],
                      out_port=config['out_port'],
                      host=config['address'],
                      device=DEVICE)
    try:
        session.start()
    except KeyboardInterrupt:
        Console.action("\nClosing session...", italic=True)
        return
