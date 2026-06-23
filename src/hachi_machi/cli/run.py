
import click
from ..osc.inference import InferenceSession
from ..console import Console
from .middleware import ClickMiddleware as M


@click.command(context_settings={'show_default': True})
@click.argument('models',
                nargs=-1,
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,)
                )
@click.option('--in-port',
              default=8000,
              help='Input OSC port.',
              type=click.IntRange(1000, 9999))
@click.option('--out-port',
              default=9000,
              help='Output OSC port.',
              type=click.IntRange(1000, 9999))
@click.option('--address', default='127.0.0.1', help='OSC IP address')
@M(path_args=[('models', '.pt'),],
   device='cpu').wrapper
def run(**config):
    """Given a path to one or more pre-trained (`.pt`) **MODELS**, starts a sessions for real-time inference, via OSC.
    Each model can be individually addressed via the `<model_id>` route—e.g., the first model has ID of `1`, the second has ID of `2`, and so on.
    The following OSC routes are available:

    ### Input routes

    - `/<model_id>/input <...features>`: The current event, as a list of features, for the model to make the next prediction. 
    The number of event features must match either the masked or unmasked event size.

    - `/<model_id>/sample`: Request a random sample from the model's learned distribution, to be sent via the `/<model_id>/output` route. Useful for kicking off an auto-regressive loop. 

    - `/<model_id>/reset`: Resets the model's hidden state (i.e., it's _context_), as well as cancelling any scheduled predictions, if the model is temporal.

    - `/stop`: Shuts down OSC server and stops session.

    :::info
    For input routes, `/<model_id>` can be optionally ommitted as a way to address all models at once.
    :::

    ### Output routes

    - `/<model_id>/output <...features>`: Model's predicted event (_unmasked_). The predicted event can be sent _as is_ back to `/input` for auto-regression, even if the model is trained on masked features.

    - `/status <int>`: OSC server status. `1` when it's launched, and `0` when closed.

    :::tip
    Note that, in some cases, running the model on CPU results in lower prediction latency.
    :::
    """
    model = config['models']
    if len(model) == 0:
        raise ValueError("You must provide at least one model")
    device = config['device']
    session = InferenceSession(models=model,
                               in_port=config['in_port'],
                               out_port=config['out_port'],
                               host=config['address'],
                               device=device)
    try:
        session.start()
    except KeyboardInterrupt:
        Console.action("\nClosing session...", italic=True)
        session.handle_stop()
