from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import os
import time
import threading
import torch
from .nn import MultiplayerAgent
from .utils import safe_handler
from .console import Console


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
        self.input_size = self.model.input_size - 2
        self.model.eval()
        self.client = SimpleUDPClient(host, out_port)
        self.dispatcher = Dispatcher()
        self.in_port = in_port
        self.out_port = out_port
        self._lock = threading.Lock()

        self._last_voice_time: dict[int, float] = {}
        self._last_time: float | None = None

        for handler in self.get_handlers():
            self.dispatcher.map(**handler)

    def get_handlers(self):
        input_size = self.model.input_layer.mean.size(-1) - 2

        @safe_handler
        def handle_input(_, *args):
            if len(args) != input_size:
                raise ValueError(
                    f"Invalid input length: {len(args)}. Expected: {input_size}")

            x = torch.tensor(
                [0.0, 0.0, *args],
                dtype=torch.float32
            )
            self.predict(x)

        @safe_handler
        def handle_temp(_, *args):
            self.model.set_temp(args[0])
            Console.action(f"Temperature: {args[0]:.3f}", italic=True)

        @safe_handler
        def handle_alpha(_, *args):
            self.model.set_alpha(args[0])
            Console.action(f"Alpha: {args[0]:.3f}", italic=True)

        @safe_handler
        def handle_reset(_):
            with self._lock:
                self.model.clear_hidden()
                self._last_voice_time.clear()
                self._last_time = None
            Console.action("Hidden state reset.", italic=True)

        @safe_handler
        def handle_weights(_, *args):
            if len(args) != self.model.input_size:
                raise ValueError(
                    f"Invalid weight size. Expected: {self.model.input_size}")
            with self._lock:
                Console.action("Setting weights", italic=True)
                self.model.set_weights(torch.tensor(args).clip(0, 1))

        return [
            {
                'address': f'/{func.__name__}',
                'handler': func
            } for func in [handle_input,
                           handle_temp,
                           handle_alpha,
                           handle_reset,
                           handle_weights]]

    def _get_voice_ioi(self, now: float, voice: int | None = None) -> float:
        return (now - self._last_voice_time[voice]) * 1000 if voice in self._last_voice_time else 0

    def _get_ioi(self, now: float) -> float:
        return (now - self._last_time) * 1000 if self._last_time is not None else 0

    def _update_time(self, voice: int, now: float) -> None:
        self._last_time = now
        self._last_voice_time[voice] = now

    def schedule(self, event: torch.Tensor, delay: float) -> None:
        def emit():
            out = event.tolist()
            msg = out[2:]
            self.client.send_message("/output", msg)
            voice = msg[0]
            if voice not in self.model.player_voices:
                self.predict(event[..., :-1])

        if delay < 33:
            emit()
        else:
            t = threading.Timer(delay / 1000.0, emit)
            t.daemon = True
            t.start()

    def predict(self, x: torch.Tensor) -> None:
        with self._lock:
            now = time.perf_counter()
            voice = int(x[..., 2].item())
            ioi = self._get_ioi(now)
            voice_ioi = self._get_voice_ioi(now, voice)
            self._update_time(voice, now)

            x = x.clone()
            x[..., 0] = ioi
            x[..., 1] = voice_ioi

            x = x.unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                y: torch.Tensor | None = self.model.forward(x)
            if y is None:
                return
            event = y.squeeze()

        delay = event[0].item()
        self.schedule(event, delay)

    def start(self):
        server = BlockingOSCUDPServer(
            server_address=(self.host, self.in_port),
            dispatcher=self.dispatcher
        )
        server.serve_forever()
