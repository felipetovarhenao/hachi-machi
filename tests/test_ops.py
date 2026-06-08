from hachi_machi.features import FeatureMap
from hachi_machi.ops import DataOperator
import torch

torch.manual_seed(1)


class TestOp:

    def setup_method(self):
        self.data = torch.rand(3, 2, dtype=torch.float32, device='cpu')
        self.fm = FeatureMap(data=self.data,
                             features={},
                             temporal=False)

    def _apply(self, *callbacks: str):
        op = DataOperator.from_callbacks(callbacks=callbacks,
                                         feature_map=self.fm)
        return op(self.data)

    def test_basic_arithmetic(self):
        y1 = self._apply('add(value=1)',
                         'mul(value=2)',
                         'sub(value=3)',
                         'div(value=4)')
        y1 = (y1 * 4 + 3) / 2 - 1
        assert torch.allclose(y1, self.data)

        dim = 0
        y2 = self._apply(f'add(value=1, dims={dim})',
                         f'mul(value=2, dims={dim})',
                         f'sub(value=3, dims={dim})',
                         f'div(value=4, dims={dim})')
        y2[..., dim] = (y2[..., dim] * 4 + 3) / 2 - 1
        assert torch.allclose(y2, self.data)

    def test_axis(self):
        eps = 0.00001
        y1 = self._apply('addrand(range=(-1, 1), axis=none)')
        assert (self.data - y1).std() < eps
        diff = self.data - y1

        y2 = self._apply('addrand(range=(-1, 1), axis=time)')
        for col in (self.data - y2).std(dim=1):
            assert col < eps

        y3 = self._apply('addrand(range=(-1, 1), axis=feature)')
        for col in (self.data - y3).std(dim=0):
            assert col < eps

        y3 = self._apply('addrand(range=(-1, 1), axis=element)')
        diff = self.data - y3
        assert diff[0, 0] != diff[0, 1] != diff[1, 0] != diff[1, 1]
