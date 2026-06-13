import time
import torch
from ..console import Console
from ..io import FileIO
from .base import BaseSession


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
