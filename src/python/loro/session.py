import click
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
        self.model: MusicAgent = torch.load(f=model,
                                            map_location=device,
                                            weights_only=False)
        self.model.eval()
        self.hidden_state = None
        self.next_hidden = None
        self.expected_event = None
        self.temp = 1.0
        self.alpha = 1.25
        self.client = SimpleUDPClient(host, out_port)
        self.dispatcher = Dispatcher()
        self.in_port = in_port
        self.weights = None
        self.weights_sum = None
        self.set_weights(torch.ones(self.model.input_size))
        for handler in self.get_handlers():
            self.dispatcher.map(**handler)

    def set_weights(self, x: torch.Tensor) -> None:
        self.weights = x.to(self.device)
        self.weights /= self.weights.sum()

    def get_confidence(self, x: torch.Tensor) -> float:
        y: torch.Tensor = (x - self.expected_event) ** 2 * self.weights
        y = y.sum().sqrt()
        return torch.exp(-self.alpha * y).item()

    def get_handlers(self):
        def handle_weights(_, *args):
            if len(args) != self.model.input_size:
                echo(
                    f"Invalid weight size. Expected: {self.model.input_size}")
                return
            self.model.set_weights(torch.tensor(args).clip(0, 1))
            echo(*self.model.weights.tolist())

        def handle_temp(_, *args):
            temp = args[0]
            self.model.set_temp(temp)
            echo(f"Temperature: {temp}")

        def handle_alpha(_, *args):
            alpha = args[0]
            self.model.set_alpha(alpha)
            echo(f"Alpha: {alpha}")

        def handle_reset(_):
            self.model.clear_hidden()
            echo(f"Clearing hidden state.")

        def handle_input(_, *args):
            if len(args) != self.model.input_size:
                echo(
                    text=f"Invalid input length: {len(args)}. Expected: {self.model.input_size}",
                    type='error')
                return
            with torch.no_grad():
                x = torch.tensor([[args]], dtype=torch.float32).to(self.device)
                y: torch.Tensor | None = self.model(x)
                if y is None:
                    return
            self.client.send_message("/output", y.cpu().tolist())

        return [
            {"address": "/temp", "handler": handle_temp},
            {"address": "/reset", "handler": handle_reset},
            {"address": "/input", "handler": handle_input},
            {"address": "/alpha", "handler": handle_alpha},
            {"address": "/weights", "handler": handle_weights},
        ]

    def start(self):
        server = BlockingOSCUDPServer(server_address=(self.host, self.in_port),
                                      dispatcher=self.dispatcher)
        echo(f"Listening on {self.host}:{self.in_port}", 'info')
        server.serve_forever()
