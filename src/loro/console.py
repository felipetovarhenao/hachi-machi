import click


class _Display:
    def __init__(self, n_rows: int = 1):
        self._first = True
        self.n_rows = n_rows
        self._fields = {}
        print()

    def _clean_key(self, key: str) -> str:
        if key not in self._fields:
            klen = len(key)
            clean = key.replace("_", " ")
            clean += ":" + " " * max(1, 17 - klen)
            self._fields[key] = f"- {clean}"
        return self._fields[key]

    def update(self, **metrics):
        if not self._first:
            print(f"\033[{self.n_rows}A", end="")
        for key, value in metrics.items():
            key = click.style(self._clean_key(key),  fg=Console.INFO)
            value = click.style(value,  fg=Console.NEUTRAL)
            print(f"\033[2K{key}{value}")
        self._first = False


class Console:

    ERROR = (223, 39, 61)
    SUCCESS = (87, 217, 139)
    NEUTRAL = (209, 223, 228)
    INFO = (180, 219, 251)
    WARNING = (233, 222, 117)
    ACTION = (242, 199, 149)

    @classmethod
    def get_display(cls, n_rows: int = 1):
        return _Display(n_rows)

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
