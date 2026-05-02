from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import os
import time
import threading
import torch
from .nn import MultiplayerAgent
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
        self.model: MultiplayerAgent = torch.load(f=model,
                                                  map_location=device,
                                                  weights_only=False)
        self.classes = set(self.model.input_layer.layers[-2].get_buffer(
            'classes_0').clone().int().tolist())
        self.input_classes = set(list(self.classes)[:1])
        self.input_size = self.model.input_size - 1
        self.model.eval()
        self.client = SimpleUDPClient(host, out_port)
        self.dispatcher = Dispatcher()
        self.in_port = in_port
        self.out_port = out_port
        self._lock = threading.RLock()
        self._last_time: float | None = None
        self._set_handlers()

    def is_input_class(self, i: int):
        return self.input_classes.issubset([i])

    def set_input_classes(self, classes: set):
        if not self.classes.issubset(classes):
            raise ValueError(f"{classes} is not a subset of {self.classes}")
        self.input_classes = classes

    def safe_handler(self, func: Callable[[str, Any], None]) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
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

    def schedule(self, event: torch.Tensor, delay: float) -> None:
        def emit():
            out = event.tolist()
            msg = out[1:]
            self.client.send_message("/output", msg)
            if self.is_input_class(msg[0]):
                self.predict(event[..., self.model.input_mask])

        t = threading.Timer(delay / 1000.0, emit)
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
            if self.is_input_class(y[..., 1].item()):
                return
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
        _, *args = args
        class_id = args[:1]
        if len(args) != self.input_size:
            raise ValueError(
                f"Invalid input length: {len(args)}. Expected: {self.input_size}")
        elif not self.classes.issuperset(class_id):
            raise ValueError(
                f"{args[0]} is not a valid class for this model. Expected i ∈ {self.classes}")
        elif not self.input_classes.issuperset(class_id):
            raise ValueError(
                f"{args[0]} has not been set as a input class. Expected i ∈ {self.input_classes}")

        x = torch.tensor(
            [0.0, *args],
            dtype=torch.float32
        )
        self.predict(x)

    def handle_reset(self, *_):
        with self._lock:
            self.model.reset()
            self._last_time = None
        Console.action("Hidden state reset.", italic=True)
