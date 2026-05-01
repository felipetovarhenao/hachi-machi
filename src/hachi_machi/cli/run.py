
import click
from ..session import Session
from ..console import Console
from .middleware import ClickMiddleware as M


@click.command(context_settings={'show_default': True})
@click.argument('model',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,)
                )
@click.option('--in-port', default=8000, help='Input OSC port.')
@click.option('--out-port', default=9000, help='Output OSC port.')
@click.option('--address', default='127.0.0.1', help='OSC IP address')
@click.option('--players', '-p',
              default=[0],
              type=int,
              help='Player indices.',
              multiple=True)
@M(path_args=[('model', '.pt'),],
   device='cpu').wrapper
def run(**config):
    """Runs a pre-trained (`.pt`) model for real-time inference, via OSC.
Note that, in some cases, running the model on CPU results in lower latency.

Arguments:

MODEL: Path to pre-trained PyTorch model (.pt)"""
    model = config['model']
    device = config['device']
    session = Session(model=model,
                      in_port=config['in_port'],
                      out_port=config['out_port'],
                      host=config['address'],
                      device=device)
    session.model.set_players(config['players'])
    try:
        session.start()
    except KeyboardInterrupt:
        Console.action("\nClosing session...", italic=True)
        return
