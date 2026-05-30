import click
from .middleware import ClickMiddleware as M
from ..io import FileIO
from ..session import RecordingSession


@click.command(context_settings={'show_default': True})
@click.argument('size',
                type=int,
                required=True)
@click.argument('output',
                default='recording.csv',
                type=click.Path(exists=False,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.option('--temporal', '-t',
              type=bool,
              help='If true, output dataset has temporal information',
              default=False,
              flag_value=True)
@click.option('--in-port', default=8000, help='Input OSC port.')
@click.option('--address', default='127.0.0.1', help='OSC IP address')
@click.option('--masked', '-m',
              type=int,
              default=[],
              multiple=True,
              help='Dimensions of masked features.')
@click.option('--categorical', '-c',
              type=int,
              default=[],
              multiple=True,
              help='Dimensions of categorical features.')
@M(path_args=[('output', *FileIO.EXT),],
   device='cpu').wrapper
def rec(**params):
    """Given a pre-defined feature size, records incoming data via OSC to create datasets in real-time. 
    **OUTPUT** file can be `.csv`, `.txt`, or `.json`.

    ### Input routes

    - `/input <list>`: Event to record. The number of event elements must match the pre-defined feature size.

    - `/stop`: Writes recorded data to disk, and stops the recording session.

    - `/reset`: Clears recorded events.
    """
    path = params['output']
    size = params['size']
    temporal = params['temporal']
    in_port = params['in_port']
    device = params['device']
    addr = params['address']
    cat = params['categorical']
    masked = params['masked']
    features = {}
    for dims, key in zip([cat, masked], ["categorical", "masked"]):
        for dim in dims:
            if dim < 0 or dim >= size:
                raise IndexError(
                    f"Dimension {dim} outside of specified size range. Should be: 0 <= dim < {size}")
            if dim not in features:
                features[dim] = {}
            features[dim][key] = True

    s = RecordingSession(path=path,
                         feature_size=size,
                         features=features,
                         temporal=temporal,
                         in_port=in_port,
                         host=addr,
                         device=device)

    s.start()
