from click.testing import CliRunner
from hachi_machi.io import FileIO
import torch


class TestFileIO:
    def _test_io(self, path, ext: str):
        fs = FileIO()
        device = 'cpu'
        tensor = torch.rand(5, 2, dtype=torch.float32, device=device)
        for i in range(2):
            temporal = bool(i)
            file_path = path / f'tmp.{ext}'
            fs.write(tensor=tensor.clone(),
                     path=file_path,
                     temporal=temporal)
            data, _ = fs.read(path=file_path, device=device)
            assert torch.allclose(tensor, data)

    def test_llll(self, tmp_path):
        self._test_io(tmp_path, 'llll')

    def test_csv(self, tmp_path):
        self._test_io(tmp_path, 'csv')
    
    def test_json(self, tmp_path):
        self._test_io(tmp_path, 'json')
    
    def test_txt(self, tmp_path):
        self._test_io(tmp_path, 'txt')
