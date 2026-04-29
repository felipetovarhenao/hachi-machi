import click
import torch
from ..augment import MidiAugmentator
from ..midi import MidiParser
from ..data import EventDataset
from ..nn import RecurrentMDN, MultiplayerAgent
from ..nn import transforms as T
from ..trainer import Trainer
from ..console import Console
from .config import Config


@click.command()
@click.argument('input',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.argument('output',
                default='model.pt',
                type=click.Path(file_okay=True, dir_okay=False))
@click.option('--mixtures',
              default=10,
              type=int,
              help='Number of Gaussian mixtures in the model.')
@click.option('--layers',
              default=1,
              help='Number of recurrent layers.',
              type=int)
@click.option('--hidden-size',
              default=256,
              type=int,
              help='Number of dimensions to use for hidden representation.')
@click.option('--context',
              default=200,
              type=int,
              help='Length of sequence segments to use during training.')
@click.option('--split',
              default=0.7,
              type=float,
              help='Training split factor.')
@click.option('--epochs',
              default=1000,
              help='Maximum number of epochs.')
@click.option('--batch-size',
              default=32,
              help='Batch size.')
@click.option('--lr',
              default=0.0025,
              help='Learning rate.')
@click.option('--patience',
              default=15,
              help='Number of iterations the model is allowed to not improve before stopping training.')
@click.option('--dropout',
              default=0.25,
              help='Dropout rate.')
@click.option('--betas',
              default=[0.9, 0.99],
              help='Betas for AdamW (Adaptive Moment Estimation) optimizer.',
              type=click.FloatRange(0.1, 0.995),
              nargs=2)
@click.option('--slope',
              default=1e-5,
              help='Negative slope for Leaky ReLU activations.')
@click.option('--seed',
              default=1,
              help='Random seed.')
@click.option('--transform', '-t',
              default=MidiAugmentator.options(),
              type=click.Choice(MidiAugmentator.options()),
              help='Data augmentation transform to randomly apply during training.',
              multiple=True)
@Config([
    ('input', '.mid', '.midi'),
    ('output', '.pt')
]).parse
def train(**params):
    """INPUT: Path to MIDI file to use as training data

    OUTPUT: Output path for trained Pytorch model (.pt)
    """
    device = params['device']
    seed = params['seed']
    midi_file = params['input']
    if seed != 0:
        torch.manual_seed(params['seed'])

    Console.action("\nParsing MIDI...", italic=True)

    parser = MidiParser(midi_file)
    num_voices = parser.numvoices()
    if parser.numvoices() < num_voices:
        raise RuntimeError(
            "MIDI file must contain of two or more channels, one for each player.")
    data = parser.events().to(device)

    input_data = data[..., :-1]
    output_data = data[...]

    IOI_DIM, VOICE_DIM = 0, 1

    # pre/post-processing layers
    input_layer = T.Transform([
        T.TimePhase(dim=IOI_DIM),
        # T.LogSpace(dims=[IOI_DIM]),
        T.Categorical(dim=VOICE_DIM, size=num_voices),
        T.Normalize(size=input_data.size(-1) + 2 + (num_voices - 1))
    ]).to(device)
    output_layer = T.Transform([
        # T.LogSpace(dims=[IOI_DIM,  -1]),
        T.Categorical(dim=VOICE_DIM, size=num_voices),
        T.Normalize(size=output_data.size(-1) + (num_voices - 1))
    ]).to(device)

    # fit to data
    input_layer.fit(input_data)
    output_layer.fit(output_data)

    augmentator = MidiAugmentator(num_voices=parser.numvoices(),
                                  transforms=params['transform'])
    dataset = EventDataset(data=data,
                           context_length=params['context'],
                           split=params['split'],
                           augmentator=augmentator)
    rnn = RecurrentMDN(k=params['mixtures'],
                       input_size=input_layer.output_size,
                       output_size=output_layer.output_size,
                       num_layers=params['layers'],
                       dropout=params['dropout'],
                       slope=params['slope'],
                       device=device)
    model = MultiplayerAgent(model=rnn,
                             input_layer=input_layer,
                             output_layer=output_layer,
                             num_voices=num_voices,
                             voice_dim=VOICE_DIM,
                             device=device,)
    trainer = Trainer(model=model,
                      dataset=dataset,
                      batch_size=params['batch_size'],
                      lr=params['lr'],
                      betas=tuple(params['betas']),)
    Console.action(
        f"{parser.numvoices()} players found", italic=True)
    trainer.run(file=params['output'],
                epochs=params['epochs'],
                patience=params['patience'])
