import torch
import enum


class FeatureType(enum.Enum):

    NORMAL = 0
    CLASS = 1
    TEMPORAL = 2


class FeatureMap:

    DEFAULTS = {
        '0': {'type': 'temporal'},
        '1': {'type': 'class'},
    }

    def __init__(self, data: torch.Tensor, features: dict):
        features = {str(int(k) + 2): v for (k, v) in features.items()}
        features = {
            **self.DEFAULTS,
            **features
        }
        size = data.size(-1)
        self.dims = list(range(size))
        self.mask = []
        self.types = []
        default_type = FeatureType.NORMAL.value
        for i in range(size):
            k = str(i)
            if k not in features:
                self.mask.append(1)
                self.types.append(default_type)
                continue
            feature: dict = features[k]
            mask = 1 - int(feature.get('masked', False))
            type = feature.get('type', 'normal')
            self.mask.append(mask)
            self.types.append(self.type_to_int(type))
        self.dims: torch.Tensor = torch.tensor(self.dims, dtype=torch.int)
        self.mask: torch.Tensor = torch.tensor(self.mask, dtype=torch.bool)
        self.types: torch.Tensor = torch.tensor(self.types, dtype=torch.int)

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
