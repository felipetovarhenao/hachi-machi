import re
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
    """Prints information for some pre-trained model.

    Arguments:

    INPUT: Path to pre-trained (.pt) model"""
    input = params['input']
    device = params['device']
    model: PerformerModel = torch.load(f=input,
                                       map_location=device,
                                       weights_only=False)
    model = model.to(device)
    Trainer.benchmark(model)
    classes = model.input_layer.layers[-2].get_buffer(
        'classes_0').int().tolist()
    Console.pretty({
        'classes': ' '.join([str(c) for c in classes]),
    }, header="info")

    graph = f"{model}"
    graph = re.sub(pattern=r"([A-Z][A-Za-z]*)(?=\()",
                   repl=Console.style(r"\g<1>", 'info', bold=True),
                   string=graph,
                   flags=re.DOTALL)
    Console.print("\nGraph", bold=True)
    Console.print(graph)
