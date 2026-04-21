import click
import torch
from .augment import MidiAugmentator
from .midi import MidiParser
from .dataset import EventDataset
from .model import FeatureScaler, RecurrentMDN, MusicAgent
from .pipeline import Pipeline
from .session import Session
from .console import Console
from .utils import (validate_path,
                    load_config,
                    DEVICE,)


@click.group()
def main():
    pass


@click.command()
@click.argument('input',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.option('--output', '-o',
              default='model.pt',
              help='Output path for trained Pytorch model (.pt).',
              type=click.Path(file_okay=True, dir_okay=False))
@click.option('--mixtures',
              default=10,
              help='Number of Gaussian mixtures in the model.')
@click.option('--hidden-size',
              default=256,
              help='Hidden size.')
@click.option('--context',
              default=200,
              help='Length of sequence segments to use during training.')
@click.option('--split',
              default=0.7,
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
              help='Betas for AdamW (Adaptive Moment Estimation) optimizer.', multiple=True)
@click.option('--slope',
              default=0.001,
              help='Negative slope for Leaky ReLU activations.')
@click.option('--seed',
              default=1,
              help='Random seed.')
def train(input, **kwargs):
    global DEVICE
    if input.endswith('.json'):
        config = load_config(input)
        midi_file = config.pop('input')
    else:
        midi_file = input
        config = {}
    params = {**kwargs, **config}

    Console.info("Training Settings:")
    for (key, value) in {"input": input, **params}.items():
        key = " ".join(key.split('_'))
        key = f"- {key}: "
        key += " " * (16 - len(key))
        Console.info(f"{key}{str(value)}")

    torch.manual_seed(params['seed'])

    Console.action("\nParsing MIDI...")
    parser = MidiParser(midi_file)
    if parser.numvoices() < 2:
        raise RuntimeError(
            "MIDI file must contain of two or more channels, one for each player.")
    data = parser.events().to(DEVICE)
    scaler = FeatureScaler(data)
    augmentator = MidiAugmentator(num_voices=parser.numvoices())
    dataset = EventDataset(data=data,
                           context_length=params['context'],
                           split=params['split'],
                           augmentator=augmentator)
    model = RecurrentMDN(k=params['mixtures'],
                         input_size=dataset.dims,
                         dropout=params['dropout'],
                         slope=params['slope'],
                         device=DEVICE)
    pipeline = Pipeline(model=model,
                        scaler=scaler,
                        dataset=dataset,
                        batch_size=params['batch_size'],
                        lr=params['lr'],
                        betas=tuple(params['betas']),)
    Console.action(
        f"Training multiplayer agent ({parser.numvoices()} players found)")
    pipeline.run(file=params['output'],
                 epochs=params['epochs'],
                 patience=params['patience'])


@click.command()
@click.argument('input')
@click.option('--in-port', default=8000, help='Input OSC port.')
@click.option('--out-port', default=9000, help='Output OSC port.')
@click.option('--address', default='127.0.0.1', help='OSC address')
def run(input, **kwargs):
    global DEVICE
    config = kwargs
    model = validate_path(input, '.pt')
    session = Session(model=model,
                      in_port=config['in_port'],
                      out_port=config['out_port'],
                      host=config['address'],
                      device=DEVICE)
    try:
        session.start()
    except KeyboardInterrupt:
        Console.action("\nClosing session...")
        return


@click.command()
@click.argument('input')
@click.argument('output')
def parse_midi(input, output):
    input = validate_path(input, ['.mid', '.midi'])
    output = validate_path(output, ['.mid', '.midi', '.txt'])

    midi = MidiParser(file=input)
    events = midi.events()
    # aug = MidiAugmentator(midi.numvoices())
    # events = aug.use_shuffle_chord(events)
    is_txt = output.endswith('.txt')
    if not is_txt:
        midi.serialize(events, output)
    else:
        out = ""
        for e in events.int().tolist():
            out += f'[ {" ".join(str(i) for i in e)} ]\n'
        with open(output, 'w') as f:
            f.write(out)


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument('output', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--tokens',  default=100, help='Number of tokens to generate.')
@click.option('--seed', default=0, help='Random seed.')
@click.option('--temp', default=1, help='Temperature.')
def generate(model_path, output, **kwargs):
    global DEVICE
    model_path = validate_path(model_path, '.pt')
    output = validate_path(output, ['.mid', '.midi', '.txt'])
    is_txt = output.endswith('.txt')
    if not is_txt:
        return
    seed = kwargs['seed']
    if seed != 0:
        torch.manual_seed(kwargs['seed'])
    agent: MusicAgent = torch.load(f=model_path,
                                   weights_only=False,
                                   map_location=DEVICE)
    model = agent.model
    scaler = agent.scaler
    model.eval()
    hidden = None

    events = []
    with torch.no_grad():
        x = torch.randn(1, 1, model.input_size, device=DEVICE)
        x = scaler(x, inverse=True)
        for _ in range(kwargs['tokens']):
            y, hidden = model.step(x=scaler(x),
                                   hidden=hidden,
                                   temp=kwargs['temp'])
            x: torch.Tensor = scaler(y.clone(), inverse=True)
            x = x.clip(0).round()
            events.append(x.squeeze())

    events = torch.stack(events).float()

    if is_txt:
        out = ""
        for e in events.int().tolist():
            out += f'[ {" ".join(str(i) for i in e)} ]\n'
        with open(output, 'w') as f:
            f.write(out)
    else:
        return
    Console.info(f"Generated {kwargs['tokens']} tokens -> {output}")


main.add_command(train)
main.add_command(run)
main.add_command(parse_midi)
main.add_command(generate)
