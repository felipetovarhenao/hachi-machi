
import click
from .middleware import ClickMiddleware as M
from ..console import Console


@click.command()
def devices():
    """Lists all available computation devices."""
    Console.info(
        '\n'.join(x for x in M.get_available_devices() if x != 'auto'))
