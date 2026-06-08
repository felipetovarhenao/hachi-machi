from hachi_machi.io import FileIO
import torch

torch.manual_seed(1)


class TestFileIO:

    def _test_io(self, path, ext: str):
        fs = FileIO()
        device = 'cpu'
        for i in range(2):
            tensor = torch.randn(16, 4, dtype=torch.float32, device=device)
            temporal = bool(i)
            file_path = path / f'tmp_{i}.{ext}'
            fs.write(tensor=tensor,
                     path=file_path,
                     temporal=temporal)
            data, fm = fs.read(path=file_path, device=device)
            assert fm.temporal() == temporal
            assert torch.allclose(tensor, data)

    def test_llll(self, tmp_path):
        self._test_io(tmp_path, 'llll')

    def test_csv(self, tmp_path):
        self._test_io(tmp_path, 'csv')

    def test_json(self, tmp_path):
        self._test_io(tmp_path, 'json')

    def test_txt(self, tmp_path):
        self._test_io(tmp_path, 'txt')
