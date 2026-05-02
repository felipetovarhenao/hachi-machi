import click
import datetime
from ..console import Console
from . import (generate,
               train,
               run,
               render, 
               train_custom,
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
main.add_command(train_custom.train_custom)
main.add_command(run.run)
main.add_command(generate.generate)
main.add_command(render.render)
main.add_command(info.info)
