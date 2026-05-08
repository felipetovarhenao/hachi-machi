import click
import datetime
from ..console import Console
from . import (augment, gen,
               format,
               train,
               run,
               info)

__banner__ = f"""
               v%(version)s 

  ██╗  ██╗ █████╗  ██████╗██╗  ██╗██╗
  ██║  ██║██╔══██╗██╔════╝██║  ██║██║
  ███████║███████║██║     ███████║██║
  ██╔══██║██╔══██║██║     ██╔══██║██║
  ██║  ██║██║  ██║╚██████╗██║  ██║██║
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝
              m a c h i  

© {datetime.datetime.now().year} https://felipe-tovar-henao.com"""


@click.version_option(message=Console.style(__banner__, 'success'))
@click.group()
def main():
    pass


main.add_command(train.train)
main.add_command(run.run)
main.add_command(gen.gen)
main.add_command(augment.augment)
main.add_command(info.info)
main.add_command(format.format)
