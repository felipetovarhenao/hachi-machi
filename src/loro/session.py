import os
import time
import threading
import torch
from .model import MusicAgent
from .utils import echo
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


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

    def _get_voice_delta(self, now: float, voice: int | None = None) -> float:
        return (now - self._last_voice_time[voice]) * 1000 if voice in self._last_voice_time else 0

    def _get_abs_delta(self, now: float) -> float:
        return (now - self._last_global_time) * 1000 if self._last_global_time is not None else 0

    def _update_timestamps(self, voice: int, now: float) -> None:
        self._last_global_time = now
        self._last_voice_time[voice] = now

    def _schedule_emission(self, event: torch.Tensor, delay_ms: float) -> None:
        def emit():
            out = event.tolist()
            msg = out[:3] + out[5:]
            self.client.send_message("/output", msg)

            voice = int(event[0].item())
            if voice not in self.model.player_voices:
                self._predict_and_schedule(event)

        if delay_ms < 5:
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

        delay_ms = event[3].item()
        self._schedule_emission(event, delay_ms)

    def get_handlers(self):
        def handle_input(_, *args):
            if len(args) != 3:
                echo(
                    f"Invalid input length: {len(args)}. Expected: 3", type='error')
                return

            now = time.perf_counter()
            voice = int(args[0])

            with self._lock:
                global_delta = self._get_abs_delta(now)
                voice_delta = self._get_voice_delta(now, voice)
                self._update_timestamps(voice, now)

            full_event = torch.tensor(
                [args[0], args[1], args[2], global_delta, voice_delta],
                dtype=torch.float32
            )

            self._predict_and_schedule(full_event)

        def handle_temp(_, *args):
            self.model.set_temp(args[0])
            echo(f"Temperature: {args[0]}")

        def handle_alpha(_, *args):
            self.model.set_alpha(args[0])
            echo(f"Alpha: {args[0]}")

        def handle_reset(_):
            with self._lock:
                self.model.clear_hidden()
                self._last_voice_time.clear()
                self._last_global_time = None
            echo("\nClearing hidden state.")

        def handle_weights(_, *args):
            if len(args) != self.model.input_size:
                echo(f"Invalid weight size. Expected: {self.model.input_size}")
                return
            with self._lock:
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
        echo(
            f"Running model: {self.name} 🤖\nOSC input: {self.host}:{self.in_port}\nOSC output: {self.host}:{self.out_port}", 'info')
        server.serve_forever()
