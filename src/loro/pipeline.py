import torch
from .timer import Timer
from .console import Console
from torch.optim import AdamW
from .model import RecurrentMDN, FeatureScaler, MusicAgent
from .loss import NLLLoss
from .dataset import EventDataset, EventLoader
from .utils import validate_path, DEVICE


class Pipeline:

    def __init__(self,
                 model: RecurrentMDN,
                 scaler: FeatureScaler,
                 dataset: EventDataset,
                 batch_size: int = 32,
                 lr: float = 0.001,
                 betas: tuple[float, float] = (0.9, 0.99)):
        self.model = model
        self.scaler = scaler
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

    def check(self, epoch: int, train_loss: float, eval_loss: float):
        if eval_loss < self.min_loss:
            self.min_loss = eval_loss
            self.patience = 0
        else:
            self.patience += 1
        self.progress = max(self.progress, self.patience)
        stop = self.patience > self.max_patience
        percent = min(100, int(round(100 * self.progress/self.max_patience)))
        if epoch > self.max_patience and self.patience == 0:
            agent = MusicAgent(model=self.model,
                               scaler=self.scaler)
            torch.save(obj=agent,
                       f=self.file)

        self.display.update(
            time=str(self.timer),
            progress=f"{percent:d}%",
            epoch=epoch,
            train_loss=f"{train_loss:1.6f}",
            validation_loss=f"{self.min_loss:.6f}"
        )
        if stop:
            Console.success(
                f"\nEpochs:\t\t{epoch:4d}\nFinal loss:\t{self.min_loss:.6f}", bold=True)

        return stop

    def benchmark(self, n_warmup: int = 50, n_runs: int = 500):
        import time
        agent: MusicAgent = torch.load(self.file,
                                       weights_only=False,
                                       map_location=DEVICE)
        model = agent.model
        model.eval()

        sample = next(iter(self.loader))[0][:1]
        sample = self.scaler(sample)

        with torch.no_grad():
            for _ in range(n_warmup):
                model(sample)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(sample)
                times.append(time.perf_counter() - t0)

        times = torch.tensor(times) * 1000
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)

        Console.pretty({
            'total': f'{total:,}',
            'trainable': f'{trainable:,}',
        }, header="Parameters:")
        Console.pretty({
            'mean': f"{times.mean():.3f}ms",
            'std': f"{times.std():.3f}ms",
            'quantile': f"{times.quantile(0.99):.3f}ms",
            'max. rate': f"{1000/times.mean():.1f}Hz",
        }, header=f"Latency ({DEVICE}):")

    def run(self, file: str, epochs: int = 1000, patience: int = 15):
        self.file = validate_path(file, '.pt')
        self.max_patience = patience
        self.patience = 0
        self.min_loss = float('inf')
        self.display = Console.get_display(n_rows=5)
        self.timer = Timer()
        self.timer.start()
        Console.print("Training info:", bold=True)
        for epoch in range(epochs):
            self.model.train()
            self.scaler.train()
            self.dataset.train()
            train_loss = 0
            train_batches = 0
            for (x, y) in self.loader:
                x = self.scaler(x)
                y = self.scaler(y)
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
            self.scaler.eval()
            with torch.no_grad():
                eval_loss = 0
                eval_batches = 0
                for (x, y) in self.loader:
                    x = self.scaler(x)
                    y = self.scaler(y)
                    pi, mu, sigma, _ = self.model(x)
                    loss: torch.Tensor = self.loss(pi, mu, sigma, y)
                    eval_loss += loss.item()
                    eval_batches += 1
                eval_loss /= eval_batches
            if self.check(epoch=epoch,
                          train_loss=2 ** train_loss,
                          eval_loss=2 ** eval_loss):
                self.benchmark()
                break
