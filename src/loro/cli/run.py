
import click
from ..session import Session
from ..console import Console
from ..utils import (device_option,
                     clean_params)


@click.command()
@click.argument('model')
@click.option('--in-port', default=8000, help='Input OSC port.')
@click.option('--out-port', default=9000, help='Output OSC port.')
@click.option('--address', default='127.0.0.1', help='OSC address')
@device_option()
def run(**kwargs):
    config = clean_params(kwargs,
                          file_keys=[
                              ('model', ['.pt'])
                          ])
    device = config['device']
    model = config['model']
    session = Session(model=model,
                      in_port=config['in_port'],
                      out_port=config['out_port'],
                      host=config['address'],
                      device=device)
    try:
        session.start()
    except KeyboardInterrupt:
        Console.action("\nClosing session...", italic=True)
        return
