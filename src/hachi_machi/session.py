import os
import time
import threading
import torch
import traceback
import functools
from abc import ABC, abstractmethod
from typing import Callable, Any
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from .nn import PerformerModel
from .console import Console
from .io import FileIO


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
            self.dispatcher.map(address=f"/{address}", handler=handler)

    def send(self, msg) -> None:
        self.client.send_message("/output", msg)

    def start(self) -> None:
        server = BlockingOSCUDPServer(
            server_address=(self.host, self.in_port),
            dispatcher=self.dispatcher
        )
        server.serve_forever()

    @abstractmethod
    def handle_input(self, *args) -> None: ...


class Session(BaseSession):
    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        self.name = os.path.basename(model)
        self.model: PerformerModel = torch.load(f=model,
                                                map_location=self.device,
                                                weights_only=False)
        self.input_size = self.model.input_size
        self.output_size = self.model.output_layer.input_size
        self.input_mask = self.model.input_mask
        self.temporal = bool(self.model.temporal)
        if not self.temporal:
            self.input_mask += 1
        else:
            self.input_size -= 1
            self.output_size -= 1
        self.model.eval()
        self._last_time: float | None = None
        self._timers: list[threading.Timer] = []

    def schedule(self, event: list, delay: float) -> None:
        t = threading.Timer(delay, lambda: self.send(event[1:]))
        t.daemon = True
        self._timers.append(t)
        t.start()

    def predict(self, x: torch.Tensor) -> None:
        now = time.perf_counter()
        with self._lock:
            x = x.clone()
            if self.temporal:
                x[..., 0] = 0 if self._last_time is None else (
                    now - self._last_time)
                self._last_time = now
            x = x.unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                y = self.model.step(x)
            event = y.squeeze().tolist()
        if self.temporal:
            delay = event[0]
            inference_ms = time.perf_counter() - now
            self.schedule(event, max(0.0, delay - inference_ms))
        else:
            self.send(event)

    def handle_input(self, *args):
        nargs = len(args)
        if nargs not in [self.input_size, self.output_size]:
            raise ValueError(
                f"Invalid input length: {nargs}. Expected: {', or '.join([str(x) for x in list({self.input_size, self.output_size})])}")

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
            [t.cancel() for t in self._timers]
            self._timers.clear()
            self._last_time = None


class RecordingSession(BaseSession):
    def __init__(self,
                 path: str,
                 feature_size: int,
                 temporal: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.temporal = temporal
        self.path, _ = FileIO.validate_path(path)
        self._buffer: list[torch.Tensor] = []
        self._start_time: float | None = None
        self.display = Console.get_display(1)

    def handle_input(self, *args) -> None:
        if len(args) != self.feature_size:
            raise ValueError(
                f"Expected feature size {self.feature_size}, got {len(args)}"
            )
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now
        timestamp = torch.tensor([now - self._start_time], dtype=torch.float32)
        row = torch.cat([timestamp, torch.tensor(args, dtype=torch.float32)])
        with self._lock:
            self._buffer.append(row)
            self.display.update(events=len(self._buffer))

    def handle_reset(self, *_) -> None:
        with self._lock:
            self._buffer.clear()
            self._start_time = None
        Console.print("Recorder reset.")

    def handle_stop(self, *_) -> torch.Tensor | None:
        with self._lock:
            if not self._buffer:
                Console.print("Nothing recorded.")
                return None
            tensor = torch.stack(self._buffer).to(self.device)
        if not self.temporal:
            tensor = tensor[..., 1:]
        else:
            tensor[1:, 0] = tensor[..., 0].diff(dim=0)
        FileIO.write(tensor, self.path, self.temporal)
        Console.success("DONE")
        exit()
