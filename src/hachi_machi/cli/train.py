import click
import torch
from ..augment import MidiAugmentator
from ..midi import MidiParser
from ..data import EventDataset
from ..nn import RecurrentMDN, MultiplayerAgent
from ..nn import transforms as T
from ..trainer import Trainer
from ..console import Console
from .middleware import ClickMiddleware as M


@click.command()
@click.option('--transform', '-t',
              default=MidiAugmentator.options(),
              type=click.Choice(MidiAugmentator.options()),
              help='Data augmentation transform to randomly apply during training.',
              multiple=True)
@M([
    ('input', '.mid', '.midi'),
    ('output', '.pt')
]).train_wrapper
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

    IOI_DIM, VOICE_DIM, DURATION_DIM = 0, 1, -1

    factory = T.TransformFactory(voice_dim=VOICE_DIM,
                                 time_dim=IOI_DIM,
                                 num_voices=num_voices)
    transforms = params['features']
    input_layer, output_layer = factory.make(input_data=input_data,
                                             output_data=output_data,
                                             transforms=transforms)

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
                       hidden_size=params['hidden_size'],
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
