import sys
import signal
import threading
import traceback
import functools
from abc import ABC
from typing import Callable, Any
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from ..console import Console


class BaseSession(ABC):
    def __init__(self,
                 in_port: int = 8000,
                 out_port: int = 9000,
                 host: str = "127.0.0.1",
                 device: str = 'cpu'):
        self.host = host
        self.in_port = in_port
        self.out_port = out_port
        self.device = device
        self.client = SimpleUDPClient(host, out_port)
        self.dispatcher = Dispatcher()
        self._lock = threading.RLock()
        self._set_handlers()
        self.close = None

    def safe_handler(self, func: Callable[[str, Any], None]) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _, *rest = args
            with self._lock:
                try:
                    func(*rest, **kwargs)
                except Exception as e:
                    self.send(0, '/status')
                    Console.error(e.args)
        return wrapper

    def _set_handlers(self) -> None:
        for attr in dir(self):
            if not attr.startswith('handle_'):
                continue
            func: Callable = getattr(self, attr)
            address = '/'.join(func.__name__.replace('handle_',
                               '').split('__'))
            handler = lambda *args, f=func: self.safe_handler(f)(*args)
            self.dispatcher.map(address=f"/{address}", handler=handler)
        self.dispatcher.set_default_handler(
            lambda addr, *_: Console.warning(f"Invalid OSC route: {addr}")
        )

    def send(self, msg, route: str = '/output') -> None:
        self.client.send_message(route, msg)

    def start(self) -> None:
        signal.signal(signal.SIGTERM, lambda s, f: self.handle_stop())
        server = BlockingOSCUDPServer(
            server_address=(self.host, self.in_port),
            dispatcher=self.dispatcher
        )
        self.send(1, '/status')
        server.serve_forever()

    def handle_stop(self, *_) -> None:
        self.send(0, '/status')
        sys.exit(0)
