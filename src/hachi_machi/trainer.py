import time
import math
import torch
from torch.optim import AdamW
from .timer import Timer
from .console import Console
from .nn import PerformerModel
from .loss import NLLLoss
from .data import EventDataset, EventLoader
from .utils import validate_path, progress


class Trainer:

    def __init__(self,
                 model: PerformerModel,
                 dataset: EventDataset,
                 batch_size: int = 32,
                 lr: float = 0.001,
                 betas: tuple[float, float] = (0.9, 0.99)):
        self.model = model
        self.dataset = dataset
        self.file = None
        batch_size = min(batch_size, self.dataset.size // 2)
        self.loader = EventLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
        self.optim = AdamW(params=model.parameters(),
                           lr=lr,
                           betas=betas)
        self.max_patience = 0
        self.patience = 0
        self.progress = 0
        self.min_loss = float('inf')
        self.loss = NLLLoss()
        self.display = None

    def _loss(self, x) -> float:
        return 1 / (1 + math.exp(-min(x, 709)))

    def check(self, epoch: int, train_loss: float, eval_loss: float) -> bool:
        if eval_loss < self.min_loss:
            self.min_loss = eval_loss
            self.patience = 0
        else:
            self.patience += 1
        self.progress = max(self.progress, self.patience)
        stop = self.patience > self.max_patience

        if self.patience == 0:
            torch.save(obj=self.model,
                       f=self.file)

        self.display.update(
            time=str(self.timer),
            progress=progress(self.progress, self.max_patience + 1),
            epoch=epoch,
            train_loss=f"{self._loss(train_loss):1.4f}",
            validation_loss=f"{self._loss(self.min_loss):.4f}"
        )
        if stop:
            Console.success(
                f"\nEpochs:\t\t{epoch:4d}\nFinal loss:\t{self._loss(self.min_loss):.6f}", bold=True)

        return stop

    @classmethod
    def benchmark(cls, model: PerformerModel, n_warmup: int = 100, n_runs: int = 500) -> None:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            sample = torch.randn(
                (1, 1, model.input_layer.output_size)).to(device)
            sample = model.input_layer(sample, True)
            for _ in range(n_warmup):
                model(sample)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(sample)
                times.append(time.perf_counter() - t0)

        times = torch.tensor(times) * 1000
        total = sum(p.numel() for p in model.parameters())

        Console.pretty({
            'total': f'{total:,}',
        }, header="Parameters")
        Console.pretty({
            'mean': f"{times.mean():.3f}ms",
            'std': f"{times.std():.3f}ms",
            'max': f"{times.quantile(0.99):.3f}ms",
            'rate': f"{1000/times.mean():.1f}Hz",
        }, header=f"Latency ({device})")

    def run(self, file: str, epochs: int = 1000, patience: int = 15) -> None:
        self.benchmark(self.model)
        self.file = validate_path(file, '.pt')
        self.max_patience = patience
        self.patience = 0
        self.min_loss = float('inf')
        self.display = Console.get_display(n_rows=5)
        self.timer = Timer()
        self.timer.start()
        Console.print("\nTraining", bold=True)
        for epoch in range(epochs):
            self.model.train()
            self.dataset.train()
            train_loss = 0
            train_batches = 0
            for (x, y) in self.loader:
                y = self.model.output_layer(y)
                pi, mu, sigma, _ = self.model(x)
                loss: torch.Tensor = self.loss(pi, mu, sigma, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_loss += loss.item()
                train_batches += 1
            train_loss /= train_batches
            self.model.eval()
            self.dataset.eval()
            with torch.no_grad():
                eval_loss = 0
                eval_batches = 0
                for (x, y) in self.loader:
                    y = self.model.output_layer(y)
                    pi, mu, sigma, _ = self.model(x)
                    loss: torch.Tensor = self.loss(pi, mu, sigma, y)
                    eval_loss += loss.item()
                    eval_batches += 1
                eval_loss /= eval_batches
            if self.check(epoch=epoch,
                          train_loss=train_loss,
                          eval_loss=eval_loss):
                break
