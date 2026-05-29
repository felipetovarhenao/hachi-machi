import click
import torch
from .. import nn
from ..data import EventDataset
from ..nn import transforms as T
from ..trainer import Trainer
from .middleware import ClickMiddleware as M
from ..io import FileIO
from ..ops.operator import DataOperator


@click.command(context_settings={'show_default': True})
@click.argument('input',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                resolve_path=True,))
@click.argument('output',
                default='model.pt',
                type=click.Path(file_okay=True, dir_okay=False))
@click.option('--mixtures', '-m',
              default=10,
              type=int,
              help='Number of Gaussian mixture components.')
@click.option('--layers', '-l',
              default=1,
              help='Number of recurrent layers.',
              type=int)
@click.option('--hidden-size', '-hs',
              default=120,
              type=int,
              help='Number of dimensions to use for hidden representation.')
@click.option('--context', '-c',
              default=200,
              type=int,
              help='Length of sequence segments to use during training.')
@click.option('--epochs', '-e',
              default=1000,
              help='Maximum number of epochs.')
@click.option('--batch-size', '-bs',
              default=32,
              help='Batch size.')
@click.option('--lr',
              default=0.0025,
              help='Learning rate.')
@click.option('--patience', '-p',
              default=15,
              help='Number of iterations the model is allowed to not improve before stopping training.')
@click.option('--dropout',
              default=0.25,
              help='Dropout rate. During training, randomly zero some of the elements of the input data. Useful to prevent over-fitting.')
@click.option('--betas',
              default=[0.9, 0.99],
              help='Coefficients used for computing running averages of gradient and its square, via _Adaptive Moment Estimation_ (Adam) optimizer.',
              type=click.FloatRange(0.1, 0.995),
              nargs=2)
@click.option('--slope',
              default=1e-5,
              type=click.FloatRange(0, max_open=True),
              help='Negative slope for Leaky ReLU activations.')
@click.option('--noise', '-n',
              default=[0, 0],
              type=click.FloatRange(0, 1),
              nargs=2,
              help='Adaptive weight noise parameters, as a pair of _standard deviation_ and _decay factor_ values, respectively. Adds Gaussian noise to the model weights during training, to prevent overfitting.')
@M.seed(1)
# @click.option('--transforms', '-t',
#               type=click.Choice(T.TransformFactory.options()),
#               help='Optional transform layers.',
#               multiple=True)
@click.option('--operations', '-op',
              type=str,
              help='Data augmentation operation(s) to stochastically apply during training. See [operations](operations)',
              multiple=True)
@M([
    ('input', *FileIO.EXT),
    ('output', '.pt')
]).wrapper
def train(**params):
    """Given a path to an **INPUT** sequential dataset (`.csv`, `.json`, `.txt`), generates an pre-trained `.pt` **OUTPUT** model, trained on that dataset. 

    Along with all other training parameters, an optional set of data augmentation [operations](operations) can be provided to be applied in series during training to the input data.
    """
    device = params['device']
    seed = params['seed']
    file_path: str = params['input']
    operations = params['operations']

    if seed != 0:
        torch.manual_seed(params['seed'])

    augmenter = None

    data, feature_map = FileIO.read(file_path, device)

    if len(operations) > 0:
        augmenter = DataOperator.from_callbacks(callbacks=operations,
                                                feature_map=feature_map)

    factory = T.TransformFactory(feature_map=feature_map)
    input_layer, output_layer = factory.make(data=data,
                                             #   transforms=params['transforms']
                                             )
    dataset = EventDataset(data=data,
                           input_dims=feature_map.input_dims(),
                           output_dims=feature_map.output_dims(),
                           context_length=params['context'],
                           augmenter=augmenter)
    rnn = nn.RecurrentMDN(k=params['mixtures'],
                          input_size=input_layer.output_size,
                          output_size=output_layer.output_size,
                          num_layers=params['layers'],
                          hidden_size=params['hidden_size'],
                          dropout=params['dropout'],
                          slope=params['slope'],
                          device=device)
    model = nn.PerformerModel(rnn=rnn,
                              input_layer=input_layer,
                              output_layer=output_layer,
                              input_mask=feature_map.input_dims(),
                              temporal=feature_map.temporal(),
                              device=device,)
    trainer = Trainer(model=model,
                      dataset=dataset,
                      batch_size=params['batch_size'],
                      lr=params['lr'],
                      betas=tuple(params['betas']),
                      adaptive_noise=tuple(params['noise']))
    trainer.run(file=params['output'],
                epochs=params['epochs'],
                patience=params['patience'])
