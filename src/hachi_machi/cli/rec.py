import click
from .middleware import ClickMiddleware as M
from ..io import FileIO
from ..console import Console
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
@M(path_args=[('output', *FileIO.EXT),],
   device='cpu').wrapper
def rec(**params):
    """Given a pre-defined feature size, records incoming data via OSC to create datasets in real-time. 
    **OUTPUT** file can be `.csv`, `.txt`, or `.json`.

    ### Input routes

    - `/input`: Sequence event to record

    - `/stop`: Writes recorded data to disk, and stops the recording session.
    """
    path = params['output']
    size = params['size']
    temporal = params['temporal']
    in_port = params['in_port']
    device = params['device']
    addr = params['address']

    s = RecordingSession(path=path,
                         feature_size=size,
                         temporal=temporal,
                         in_port=in_port,
                         host=addr,
                         device=device)

    s.start()
