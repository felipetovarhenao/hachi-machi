import click
from ..console import Console
from . import (generate,
               train,
               run,
               render)

__banner__ = """
               v%(version)s 

  ██╗  ██╗ █████╗  ██████╗██╗  ██╗██╗
  ██║  ██║██╔══██╗██╔════╝██║  ██║██║
  ███████║███████║██║     ███████║██║
  ██╔══██║██╔══██║██║     ██╔══██║██║
  ██║  ██║██║  ██║╚██████╗██║  ██║██║
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝
              m a c h i  

© 2026 https://felipe-tovar-henao.com"""


@click.version_option(message=Console.style(__banner__, 'success'))
@click.group()
def main():
    pass


main.add_command(train.train)
main.add_command(run.run)
main.add_command(generate.generate)
main.add_command(render.render)
