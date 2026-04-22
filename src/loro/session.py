from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import os
import time
import threading
import torch
from .model import MusicAgent
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
        self.model: MusicAgent = torch.load(f=model,
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
        self._last_global_time: float | None = None

        for handler in self.get_handlers():
            self.dispatcher.map(**handler)

    def _get_voice_ioi(self, now: float, voice: int | None = None) -> float:
        return (now - self._last_voice_time[voice]) * 1000 if voice in self._last_voice_time else 0

    def _get_ioi(self, now: float) -> float:
        return (now - self._last_global_time) * 1000 if self._last_global_time is not None else 0

    def _update_timestamps(self, voice: int, now: float) -> None:
        self._last_global_time = now
        self._last_voice_time[voice] = now

    def _schedule_emission(self, event: torch.Tensor, delay_ms: float) -> None:
        def emit():
            out = event.tolist()
            msg = out[2:]
            self.client.send_message("/output", msg)
            voice = msg[0]
            if voice not in self.model.player_voices:
                self._predict_and_schedule(event)

        if delay_ms < 33:
            emit()
        else:
            t = threading.Timer(delay_ms / 1000.0, emit)
            t.daemon = True
            t.start()

    def _predict_and_schedule(self, x: torch.Tensor) -> None:
        with self._lock:
            x = x.unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                y: torch.Tensor | None = self.model(x)
            if y is None:
                return
            event = y.squeeze()

        delay_ms = event[0].item()
        self._schedule_emission(event, delay_ms)

    def get_handlers(self):
        @safe_handler
        def handle_input(_, *args):
            if len(args) != self.input_size:
                raise ValueError(
                    f"Invalid input length: {len(args)}. Expected: {self.input_size}")

            now = time.perf_counter()
            voice = int(args[0])

            with self._lock:
                ioi = self._get_ioi(now)
                voice_ioi = self._get_voice_ioi(now, voice)
                self._update_timestamps(voice, now)

            full_event = torch.tensor(
                [ioi, voice_ioi, *args],
                dtype=torch.float32
            )
            self._predict_and_schedule(full_event)

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
                self._last_global_time = None
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
            {"address": "/input",   "handler": handle_input},
            {"address": "/temp",    "handler": handle_temp},
            {"address": "/alpha",   "handler": handle_alpha},
            {"address": "/reset",   "handler": handle_reset},
            {"address": "/weights", "handler": handle_weights},
        ]

    def start(self):
        server = BlockingOSCUDPServer(
            server_address=(self.host, self.in_port),
            dispatcher=self.dispatcher
        )
        server.serve_forever()
