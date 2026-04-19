class Console:

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    @classmethod
    def info(cls, msg: str, **kwargs) -> None:
        print(f"{cls.CYAN}{msg}{cls.RESET}", **kwargs)

    @classmethod
    def success(cls, msg: str, **kwargs) -> None:
        print(f"{cls.GREEN}{msg}{cls.RESET}", **kwargs)

    @classmethod
    def warning(cls, msg: str) -> Warning:
        print(f"{cls.YELLOW}{msg}{cls.RESET}")

    @classmethod
    def error(cls, msg: str, err_cls: BaseException = Exception) -> BaseException:
        raise err_cls(f"{cls.RED}{msg}{cls.RESET}")

    @classmethod
    def action(cls, msg: str, **kwargs) -> None:
        print(f"{cls.MAGENTA}{msg}{cls.RESET}", **kwargs)
