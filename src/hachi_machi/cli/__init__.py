import click
import datetime
from .middleware import ClickMiddleware
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


@main.command()
@click.argument('input', type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def exec(ctx, input):
    params = ClickMiddleware.from_file(input)
    valid_cmds = [x for x in list(main.commands) if x != 'exec']
    options = ', '.join(valid_cmds)
    try:
        cmd_name = params.pop('cmd')
    except:
        Console.error(
            f"You must specify one of the following commands: {options}")
        exit()
    if cmd_name not in valid_cmds:
        Console.error(
            f"Invalid command: {cmd_name!r}. Expected: {options}")
        exit()
    cmd = main.commands[cmd_name]
    ctx.invoke(cmd, **params)


main.add_command(train.train)
main.add_command(run.run)
main.add_command(gen.gen)
main.add_command(augment.augment)
main.add_command(info.info)
main.add_command(format.format)
