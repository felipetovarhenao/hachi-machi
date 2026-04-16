import torch
from tqdm import tqdm
from torch.optim import AdamW
from model import EventModel, Scaler, MusicAgent
from loss import NLLLoss
from dataset import EventDataset, EventLoader


class Pipeline:

    def __init__(self,
                 model: EventModel,
                 scaler: Scaler,
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
        self.min_loss = float('inf')
        self.loss = NLLLoss()
        self.pbar: tqdm | None = None

    def check(self, epoch: int, train_loss: float, eval_loss: float):
        prev_loss = self.min_loss
        if eval_loss < self.min_loss:
            self.min_loss = eval_loss
            self.patience = 0
        else:
            self.patience += 1
        stop = self.patience > self.max_patience
        percent = min(100, int(round(100 * self.patience/self.max_patience)))
        if epoch > 5 and self.patience == 0:
            torch.save(obj=self.model,
                       f=self.file)
        self.pbar.set_postfix_str(
            f"{percent:3d}% | epoch: {epoch:4d} | T-loss: {train_loss:1.6f} | E-loss: {self.min_loss:.6f} | learning: {(prev_loss - eval_loss):.6f}")
        if stop:
            self.pbar.close()
            print(
                f"***\nEpochs: {epoch:4d}\tFinal loss: {self.min_loss:.6f}")
        return stop

    def run(self, file: str, epochs: int = 1000, patience: int = 15):
        self.file = file
        self.max_patience = patience
        self.patience = 0
        self.min_loss = float('inf')
        self.pbar = tqdm(iterable=range(epochs),
                         bar_format="[{elapsed}]{postfix}",
                         colour="#4d94c0")

        for epoch in self.pbar:
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
                break
