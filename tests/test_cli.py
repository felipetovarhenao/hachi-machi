import pytest
import torch
from hachi_machi.io import FileIO
from hachi_machi.cli import main
from click.testing import CliRunner, Result
from pathlib import Path


@pytest.fixture(scope='class')
def sample_files(tmp_path_factory):
    tmp: Path = tmp_path_factory.mktemp("data")
    x = torch.randn(10, 4)
    data = {}
    for i, name in enumerate(['temporal', 'atemporal']):
        path = tmp / f"{name}.csv"
        FileIO.write(tensor=x, path=path, temporal=bool(i))
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
