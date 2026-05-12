import torch
import enum


class FeatureType(enum.Enum):

    CONTINUOUS = 0
    CATEGORICAL = 1


class FeatureMap:

    def __init__(self, data: torch.Tensor, features: dict, temporal: bool = False):
        size = data.size(-1)
        self.dims = list(range(size))
        self.mask = []
        self._temporal = temporal
        self.types = []
        default_type = FeatureType.CONTINUOUS.value
        for i in range(size):
            k = str(i - int(temporal))
            if k not in features:
                self.mask.append(1)
                self.types.append(default_type)
                continue
            feature: dict = features[k]
            mask = 1 - int(feature.get('masked', False))
            type = int(feature.get('categorical', False))
            self.mask.append(mask)
            self.types.append(type)
        self.dims: torch.Tensor = torch.tensor(self.dims, dtype=torch.int)
        self.mask: torch.Tensor = torch.tensor(self.mask, dtype=torch.bool)
        self.types: torch.Tensor = torch.tensor(self.types, dtype=torch.int)
        if not torch.any(self.mask[int(temporal):]):
            raise RuntimeError(
                f"At least one feature must be unmasked: {features!r}")

    @staticmethod
    def type_to_int(name: str) -> int:
        return getattr(FeatureType, name.upper()).value

    def input_dims(self, name: str | None = None) -> list:
        mask = self.mask
        if name:
            mask = self.mask & (self.types == self.type_to_int(name))
        return self.dims[mask].tolist()

    def output_dims(self, name: str | None = None) -> list:
        mask = slice(None, None)
        if name:
            mask = self.types == self.type_to_int(name)
        return self.dims[mask].tolist()

    def __repr__(self):
        return str(torch.stack([self.dims, self.mask.int(), self.types]))

    def __len__(self):
        return len(self.dims)

    def temporal(self):
        return self._temporal

    def to_dict(self):
        features = {}
        for (dim, type, mask) in zip(self.dims, self.types, self.mask):
            f = {}
            if type == FeatureType.CATEGORICAL.value:
                f['categorical'] = True
            if mask == 0:
                f['masked'] = True
            id = int(dim.item())
            if self.temporal():
                id -= 1
            features[str(id)] = f
        return features
