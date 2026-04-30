import click
from ..console import Console
from . import (generate,
               train,
               run,
               render)


@click.version_option(message=Console.style("v%(version)s", 'success'))
@click.group()
def main():
    pass


main.add_command(train.train)
main.add_command(run.run)
main.add_command(generate.generate)
main.add_command(render.render)
