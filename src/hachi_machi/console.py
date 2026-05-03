import click


class _Display:
    def __init__(self, n_rows: int = 1):
        self._first = True
        self.n_rows = n_rows
        self._fields = {}

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
            key = Console.print(self._clean_key(key), defer=True)
            value = Console.info(value,  defer=True)
            print(f"\033[2K{key}{value}")
        self._first = False


class Console:

    ERROR = (250, 130, 130)
    SUCCESS = (120, 230, 189)
    NEUTRAL = (220, 220, 220)
    INFO = (160, 200, 240)
    WARNING = (233, 222, 117)
    ACTION = (242, 199, 149)

    @classmethod
    def get_display(cls, n_rows: int = 1):
        return _Display(n_rows)

    @classmethod
    def print(cls, text: str, type: str = 'neutral', defer: bool = False, end: str | None = '\n', **kwargs):
        text = cls.style(text, type, **kwargs)
        return text if defer else print(text, end=end)

    @classmethod
    def style(cls, text: str, type: str = 'neutral', **kwargs):
        return click.style(text=text, fg=getattr(cls, type.upper()), **kwargs)

    @classmethod
    def info(cls, msg: str, **kwargs) -> None:
        return cls.print(msg, 'info', **kwargs)

    @classmethod
    def action(cls, msg: str, **kwargs) -> None:
        return cls.print(msg, 'action', **kwargs)

    @classmethod
    def success(cls, msg: str, **kwargs) -> None:
        return cls.print(msg, 'success', **kwargs)

    @classmethod
    def warning(cls, msg: str, **kwargs) -> None:
        return cls.print(msg, 'warning', **kwargs)

    @classmethod
    def error(cls, msg: str, **kwargs) -> None:
        return cls.print(msg, 'error', **kwargs)

    @classmethod
    def pretty(cls, obj: dict, header: str | None = None):
        if header:
            Console.print(f"\n{header.capitalize()}", bold=True)
        col_size = 20

        for (k, v) in sorted(obj.items(), key=lambda x: x[0]):
            indent = ''
            if isinstance(v, list | tuple):
                indent = ' ' * col_size
                if len(v) > 0:
                    v = f',\n{indent}'.join([str(i) for i in v])
                else:
                    v = 'none'
            k: str = k.replace('_', " ")
            k = f"- {k}:"
            k += " " * max(1, col_size - len(k))
            k = Console.style(k)
            v = Console.info(v, defer=True)
            Console.print(f"{k}{v}")
