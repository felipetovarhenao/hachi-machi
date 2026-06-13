import time
import threading
import torch
from ..nn import PerformerModel
from .base import BaseSession


class InferenceSession(BaseSession):
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

    def schedule(self, event: list, delay: float) -> None:
        index = self.index

        def _fire():
            with self._lock:
                self.send(event[1:], f"/{index}/output")
                self.models[index]['timers'].discard(t)

        t = threading.Timer(delay, _fire)
        t.daemon = True
        with self._lock:
            self.models[index]['timers'].add(t)
        t.start()

    def predict(self, x: torch.Tensor) -> None:
        now = time.perf_counter()
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
            self.send(event, f"/{self.index}/output")

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
        self.send(y[-self.output_size:].tolist(), f"/{self.index}/output")

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
        [t.cancel() for t in self.timers]
        self.timers.clear()
        self.model.reset()
        self.last_time = None
