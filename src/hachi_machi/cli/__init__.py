import click
import datetime
import webbrowser
from .middleware import ClickMiddleware
from ..console import Console
from . import (fork, gen,
               format,
               train,
               run,
               rec,
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


class CustomGroup(click.Group):

    @staticmethod
    def open_docs(ctx, param, value):
        if not value or ctx.resilient_parsing:
            return
        webbrowser.open("http://localhost:3000/docs/")
        ctx.exit()

    @classmethod
    def help_params(cls):
        return dict(is_flag=True,
                    is_eager=True,
                    expose_value=False,
                    callback=cls.open_docs,
                    help="Open documentation in browser.")

    def add_command(self, cmd, name=None):
        cmd.params.append(
            click.Option(['-h', '--help'], **self.help_params())
        )
        super().add_command(cmd, name)


@click.version_option(message=Console.style(__banner__, 'success'))
@click.group(cls=CustomGroup)
@click.option('-h', '--help', **CustomGroup.help_params())
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
main.add_command(fork.fork)
main.add_command(info.info)
main.add_command(format.format)
main.add_command(rec.rec)
