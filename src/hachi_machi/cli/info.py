import click
import torch
from .middleware import ClickMiddleware as M
from ..console import Console
from ..nn.performer import PerformerModel
from ..trainer import Trainer


@click.command()
@click.argument('input')
@M([('input', '.pt')]).wrapper
def info(**params):
    """Prints information for some **INPUT** pre-trained model."""
    input = params['input']
    device = params['device']
    model: PerformerModel = torch.load(f=input,
                                       map_location=device,
                                       weights_only=False)
    model = model.to(device)
    Trainer.benchmark(model)
