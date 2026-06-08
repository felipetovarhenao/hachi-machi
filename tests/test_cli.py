import pytest
import torch
from hachi_machi.io import FileIO
from hachi_machi.cli import main
from hachi_machi.nn.performer import PerformerModel
from click.testing import CliRunner, Result
from pathlib import Path

torch.manual_seed(1)


@pytest.fixture(scope='class')
def sample_files(tmp_path_factory):
    tmp: Path = tmp_path_factory.mktemp("data")
    x = torch.randn(20, 4)
    features = {
        '1': {
            'masked': True
        },
    }
    data = {}
    for i, name in enumerate(['atemporal', 'temporal']):
        path = tmp / f"{name}.csv"
        FileIO.write(tensor=x,
                     path=path,
                     temporal=bool(i),
                     features=features)
        data[name] = path
    return data


class TestCli:

    def setup_method(self):
        self.cli = CliRunner()

    def _eval(self, result: Result, out_path: str | None = None):
        assert result.exit_code == 0
        if out_path:
            assert Path(out_path).exists()

    def test_format(self, sample_files):
        ds: Path = sample_files['temporal']
        in_path = str(ds.absolute())
        out_path = in_path.replace('.csv', '.llll')
        result = self.cli.invoke(main, ['fork', in_path, out_path])
        assert result.exit_code == 0
        assert Path(out_path).exists()

    def test_train(self, sample_files):
        ds: Path = sample_files['temporal']
        in_path = str(ds.absolute())
        out_path = str((ds.parent / 'model.pt').absolute())
        result = self.cli.invoke(
            main, ['train', in_path, out_path, '--epochs', 2])
        self._eval(result, out_path)
        result = self.cli.invoke(main, ['info', out_path])
        self._eval(result, out_path)
        model: PerformerModel = torch.load(f=out_path,
                                           weights_only=False,
                                           map_location='cpu')
        assert bool(model.temporal) is True
        x = torch.randn((1, 1, model.input_size))
        y = model.step(x)
        assert y.size(-1) == model.output_layer.output_size
        assert y.size(-1) != x.size(-1)
