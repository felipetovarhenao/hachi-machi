
import click
from ..session import Session
from ..console import Console
from .config import Config


@click.command()
@click.argument('model',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,)
                )
@click.option('--in-port', default=8000, help='Input OSC port.')
@click.option('--out-port', default=9000, help='Output OSC port.')
@click.option('--address', default='127.0.0.1', help='OSC IP address')
@click.option('--temp', default=1.0, type=float, help='Temperature')
@click.option('--alpha', default=1.0, type=float, help='Alpha')
@Config([
    ('model', '.pt'),
]).parse
def run(**config):
    """MODEL: Path to pre-trained PyTorch model (.pt)"""
    model = config['model']
    device = config['device']
    session = Session(model=model,
                      in_port=config['in_port'],
                      out_port=config['out_port'],
                      host=config['address'],
                      device=device)
    session.model.set_temp(config['temp'])
    session.model.set_alpha(config['alpha'])
    try:
        session.start()
    except KeyboardInterrupt:
        Console.action("\nClosing session...", italic=True)
        return
