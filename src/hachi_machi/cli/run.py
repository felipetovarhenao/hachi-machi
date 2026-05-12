
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
@M(path_args=[('model', '.pt'),],
   device='cpu').wrapper
def run(**config):
    """Given a path to pre-trained (`.pt`) **MODEL**, starts a sessions for real-time inference, via OSC.
    The following OSC routes are available:

    ### Input routes

    - `/input`: Receives an event for prediction. 
    The number of event features must match either the masked or unmasked event size.

    - `/reset`: Resets the model's hidden state (i.e., it's _context_), as well as cancelling any scheduled predictions, if the model is temporal.

    ### Output routes

    - `/output`: Predicted event (_unmasked_). The predicted event can be sent _as is_ back to `/input` for auto-regression, even if the model is trained on masked features.

    :::tip
    Note that, in some cases, running the model on CPU results in lower prediction latency.
    :::
    """
    model = config['model']
    device = config['device']
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
