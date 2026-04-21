import click


class Console:

    ERROR = (226, 45, 78)
    SUCCESS = (43, 197, 133)
    NEUTRAL = (161, 162, 164)
    INFO = (146, 176, 203)
    WARNING = (186, 172, 93)
    ACTION = (179, 123, 194)

    @classmethod
    def print(cls, text: str, type: str = 'neutral'):
        return print(cls.style(text, type))

    @classmethod
    def style(cls, text: str, type: str):
        return click.style(text, fg=getattr(cls, type.upper()))

    @classmethod
    def info(cls, msg: str) -> None:
        cls.print(msg, 'info')

    @classmethod
    def action(cls, msg: str) -> None:
        cls.print(msg, 'action')

    @classmethod
    def success(cls, msg: str) -> None:
        cls.print(msg, 'success')

    @classmethod
    def warning(cls, msg: str) -> None:
        cls.print(msg, 'warning')

    @classmethod
    def error(cls, msg: str) -> None:
        cls.print(msg, 'error')
