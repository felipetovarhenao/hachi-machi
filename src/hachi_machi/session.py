from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import os
import time
import threading
import torch
from .nn import PerformerModel
from .console import Console
import traceback
from typing import Callable, Any
import functools


class Session:
    def __init__(self,
                 model: str,
                 in_port: int = 8000,
                 out_port: int = 9000,
                 host: str = "127.0.0.1",
                 device: str = 'mps'):
        self.device = device
        self.host = host
        self.name = os.path.basename(model)
        self.model: PerformerModel = torch.load(f=model,
                                                map_location=device,
                                                weights_only=False)
        self.input_size = self.model.input_size - 1
        self.output_size = self.model.output_layer.input_size - 1
        self.model.eval()
        self.client = SimpleUDPClient(host, out_port)
        self.dispatcher = Dispatcher()
        self.in_port = in_port
        self.out_port = out_port
        self._lock = threading.RLock()
        self._last_time: float | None = None
        self.display = Console.get_display(1)
        self._set_handlers()

    def safe_handler(self, func: Callable[[str, Any], None]) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _, *rest = args
            try:
                func(*rest, **kwargs)
            except Exception:
                Console.error(traceback.format_exc())
        return wrapper

    def _set_handlers(self) -> None:
        for attr in dir(self):
            if not attr.startswith('handle_'):
                continue
            func: Callable = getattr(self, attr)
            address = func.__name__.replace('handle_', '')
            handler = lambda *args, f=func: self.safe_handler(f)(*args)
            self.dispatcher.map(address=f"/{address}",
                                handler=handler)

    def send(self, msg):
        self.client.send_message("/output", msg)
        # self.display.update(output=msg)

    def schedule(self, event: torch.Tensor, delay: float) -> None:
        t = threading.Timer(
            delay / 1000.0, lambda: self.send(event.tolist()[1:]))
        t.daemon = True
        t.start()

    def predict(self, x: torch.Tensor) -> None:
        now = time.perf_counter()
        with self._lock:
            x = x.clone()
            x[..., 0] = 0 if self._last_time is None else (
                now - self._last_time) * 1000
            self._last_time = now

            x = x.unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                y = self.model.step(x)
            event = y.squeeze()

        delay = event[0].item()
        inference_ms = (time.perf_counter() - now) * 1000
        self.schedule(event, max(0.0, delay - inference_ms))

    def start(self):
        server = BlockingOSCUDPServer(
            server_address=(self.host, self.in_port),
            dispatcher=self.dispatcher
        )
        server.serve_forever()

    def handle_input(self, *args):
        nargs = len(args)
        if nargs not in [self.input_size, self.output_size]:
            raise ValueError(
                f"Invalid input length: {nargs}. Expected: {', or '.join([str(self.input_size), str(self.output_size)])}")

        x = torch.tensor(
            [0.0, *args],
            dtype=torch.float32
        )
        if nargs == self.output_size:
            x = x[self.model.input_mask]
        self.predict(x)

    def handle_reset(self, *_):
        with self._lock:
            self.model.reset()
            self._last_time = None
        self.display.update(reset='*')
