import time
import threading
import torch
import traceback
import functools
from abc import ABC
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
            address = '/'.join(func.__name__.replace('handle_',
                               '').split('__'))
            handler = lambda *args, f=func: self.safe_handler(f)(*args)
            self.dispatcher.map(address=f"/{address}", handler=handler)
        self.dispatcher.set_default_handler(
            lambda addr, *_: Console.warning(f"{addr!r} is not a valid route.")
        )

    def send(self, msg) -> None:
        self.client.send_message("/output", msg)

    def start(self) -> None:
        server = BlockingOSCUDPServer(
            server_address=(self.host, self.in_port),
            dispatcher=self.dispatcher
        )
        server.serve_forever()


class Session(BaseSession):
    def __init__(self, models: list[str], **kwargs):
        self.models: dict = {}
        self.index = str(1)
        for i, file in enumerate(models):
            id = str(i + 1)
            model: PerformerModel = torch.load(f=file,
                                               map_location=kwargs['device'],
                                               weights_only=False)
            model.eval()
            input_size = model.input_size
            output_size = model.output_layer.input_size
            input_mask = model.input_mask
            temporal = bool(model.temporal)
            if not temporal:
                input_mask += 1
            else:
                input_size -= 1
                output_size -= 1
            self.models[id] = {
                'model': model,
                'input_size': input_size,
                'output_size': output_size,
                'temporal': temporal,
                'last_time': None,
                'timers': set()
            }

            for route in ['input', 'sample', 'reset']:
                base_method = getattr(self, f'_handle_{route}')
                if i == 0:
                    wild_route = f"handle_{route}"

                    def wildcard(*args, _base=base_method, **kwargs):
                        for id in self.models.keys():
                            self.index = id
                            _base(*args, **kwargs)
                    wildcard.__name__ = wild_route
                    setattr(self, wild_route, wildcard)

                subroute = f"handle_{id}__{route}"

                def func(*args, _id=id, _base=base_method, **kwargs):
                    self.index = _id
                    _base(*args, **kwargs)
                func.__name__ = subroute
                setattr(self, subroute, func)

        super().__init__(**kwargs)

    def send(self, msg) -> None:
        self.client.send_message(f"/{self.index}/output/", msg)

    def schedule(self, event: list, delay: float) -> None:
        def _fire():
            self.send(event[1:])
            with self._lock:
                self.timers.discard(t)

        t = threading.Timer(delay, _fire)
        t.daemon = True
        self.timers.add(t)
        t.start()

    def predict(self, x: torch.Tensor) -> None:
        now = time.perf_counter()
        with self._lock:
            x = x.clone()
            if self.temporal:
                x[..., 0] = 0 if self.last_time is None else (
                    now - self.last_time)
                self.last_time = now
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

    @property
    def current(self) -> dict:
        return self.models[self.index]

    @current.setter
    def current(self, value):
        self.models[self.index] = value

    @property
    def model(self) -> PerformerModel:
        return self.current['model']

    @property
    def input_size(self) -> int:
        return self.current['input_size']

    @property
    def output_size(self) -> int:
        return self.current['output_size']

    @property
    def timers(self) -> set[threading.Timer]:
        return self.current['timers']

    @property
    def last_time(self) -> None | float:
        return self.current['last_time']

    @last_time.setter
    def last_time(self, value: int):
        self.current['last_time'] = value

    @property
    def temporal(self) -> bool:
        return self.current['temporal']

    def _handle_sample(self, *_):
        size = self.model.output_layer.output_size
        y = self.model.output_layer(torch.randn(size), True)
        self.send(y[-self.output_size:].tolist())

    def _handle_input(self, *args):
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
        else:
            x = x[-self.model.input_size:]
        self.predict(x)

    def _handle_reset(self, *_):
        with self._lock:
            [t.cancel() for t in self.timers]
            self.timers.clear()
            self.model.reset()
            self.last_time = None


class RecordingSession(BaseSession):
    def __init__(self,
                 path: str,
                 feature_size: int,
                 features: dict,
                 temporal: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.features = features
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
        FileIO.write(tensor, self.path, self.temporal, features=self.features)
        Console.success("DONE")
        exit()
