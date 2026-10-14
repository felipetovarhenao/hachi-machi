from hachi_machi.features import FeatureMap
from hachi_machi.ops import DataOperator
import torch

torch.manual_seed(1)


class TestOp:

    def setup_method(self):
        torch.manual_seed(1)
        self.data = torch.rand(3, 2, dtype=torch.float32, device='cpu')
        self.original = self.data.clone()
        self.fm = FeatureMap(data=self.data,
                             features={},
                             temporal=False)

    def _apply(self, *callbacks: str):
        op = DataOperator.from_callbacks(callbacks=callbacks,
                                         feature_map=self.fm)
        y = op(self.data)
        assert torch.equal(self.data, self.original)
        assert y.data_ptr() != self.data.data_ptr()
        assert y.shape == self.original.shape and y.dtype == self.original.dtype
        return y

    def test_basic_arithmetic(self):
        x = self.original

        y1 = self._apply('add(value=1)',
                         'mul(value=2)',
                         'sub(value=3)',
                         'div(value=4)')
        assert torch.allclose(y1, ((x + 1) * 2 - 3) / 4, rtol=1e-6, atol=1e-6)
        assert not torch.allclose(y1, x)

        dim = 0
        other = 1 - dim
        y2 = self._apply(f'add(value=1, dims={dim})',
                         f'mul(value=2, dims={dim})',
                         f'sub(value=3, dims={dim})',
                         f'div(value=4, dims={dim})')
        expected2 = x.clone()
        expected2[..., dim] = ((x[..., dim] + 1) * 2 - 3) / 4
        assert torch.allclose(y2, expected2, rtol=1e-6, atol=1e-6)
        assert not torch.allclose(y2[..., dim], x[..., dim])
        assert torch.equal(y2[..., other], x[..., other])

    def _offset(self, axis: str) -> torch.Tensor:
        y = self._apply(f'addrand(range=(-1, 1), axis={axis})')
        d = y - self.original
        assert d.abs().max() <= 1 + 1e-6
        return d

    def test_axis(self):
        eps = 1e-5
        var = 1e-3

        d = self._offset('none')
        assert d.std() < eps
        assert d.mean().abs() > eps

        d = self._offset('time')
        assert d.std(dim=1).max() < eps
        assert d[:, 0].std() > var

        d = self._offset('feature')
        assert d.std(dim=0).max() < eps
        assert d[0, :].std() > var

        d = self._offset('element')
        assert d.flatten().unique().numel() == d.numel()
