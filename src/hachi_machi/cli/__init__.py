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
    """
    High-level and controllable human interface for machine improvisation.

    ## Usage

    ```
    hxmx <command> ...
    ```
    """
    pass


@main.command()
@click.argument('input', type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def exec(ctx, input):
    """
    Allows to run commands via YAML or TOML configuration files. 
    All configuration files must provide values for the command, via the `cmd` key, and any required positional argument.

    For instance:

    ### YAML 

    ```yaml title="config.yaml" showLineNumbers
    cmd: info
    input: ./mymodel.pt
    ```

    ### TOML

    ```toml title="config.toml" showLineNumbers
    cmd = "gen"
    input = "./mymodel.pt"
    output = "output.txt"
    tokens = 300
    ```
    """
    try:
        params = ClickMiddleware.from_file(input)
        valid_cmds = [x for x in list(main.commands) if x != 'exec']
        options = ', '.join(valid_cmds)
        cmd_name = params.pop('cmd', None)
        if cmd_name is None or cmd_name not in valid_cmds:
            raise ValueError(
                f"Invalid command: {cmd_name!r}. Expected: {options}")
        cmd = main.commands[cmd_name]
    except Exception as e:
        Console.error(e.args[0])
        exit()
    ctx.invoke(cmd, **params)


main.add_command(train.train)
main.add_command(run.run)
main.add_command(gen.gen)
main.add_command(augment.augment)
main.add_command(info.info)
main.add_command(format.format)
