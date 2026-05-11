import os
import json
import torch
import csv
from .features import FeatureMap


class FileIO:

    EXT = ('.txt', '.csv', '.json')

    @classmethod
    def validate_path(cls, file: str) -> tuple[str, str]:
        file = os.path.abspath(os.path.expanduser(file))
        file_ext = os.path.splitext(file)[1]
        if file_ext not in cls.EXT:
            raise TypeError(
                f"Invalid extension: {file_ext}. Expected: {', '.join(cls.EXT)}")
        return file, file_ext

    @classmethod
    def read(cls, path: str, device: torch.device) -> tuple[torch.Tensor, FeatureMap]:
        path, ext = cls.validate_path(path)
        reader = getattr(cls, f'read_{ext[1:]}')
        tensor, feature_map = reader(path)
        if feature_map.temporal():
            tensor[..., 0] = tensor[..., 0].diff(0)
        return tensor.to(device), feature_map

    @classmethod
    def write(cls, tensor: torch.Tensor, path: str, temporal: bool = False, **kwargs) -> None:
        assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
        path, ext = cls.validate_path(path)
        writer = getattr(cls, f'write_{ext[1:]}')
        writer(tensor, path, temporal, **kwargs)

    @classmethod
    def to_tensor(cls, data: list[list[float]]) -> torch.Tensor:
        try:
            tensor = torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Invalid data formatting:\n{e.args[0]}")
        return tensor

    @classmethod
    def write_csv(cls, tensor: torch.Tensor, path: str, temporal: bool = False, **_) -> None:
        n_features = tensor.shape[1]

        header = ["time"]
        header += [f"{i}" for i in range(n_features - int(temporal))]
        header = header[-n_features:]
        with open(path, "w") as f:
            f.write(",".join(header) + "\n")
            for row in tensor.tolist():
                f.write(",".join(map(str, row)) + "\n")

    @classmethod
    def read_csv(cls, path: str) -> tuple[torch.Tensor, FeatureMap]:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            temporal = header[0] == 'time'
            data = [[float(v) for v in row] for row in reader]
        try:
            data = cls.to_tensor(data)
        except:
            ImportError(f"Invalid CSV formatting in {path!r}")
        features = {}
        for (i, k) in enumerate(header[int(temporal):]):
            ft = {}
            if k.startswith('-'):
                ft['masked'] = True
            if k.endswith('!'):
                ft['categorical'] = True
            if len(ft) > 0:
                features[str(i)] = ft
        return data, FeatureMap(data, features, temporal)

    @classmethod
    def write_txt(cls, tensor: torch.Tensor, output: str, temporal: bool = False, **_) -> None:
        rows = [[repr(x) for x in row] for row in tensor.tolist()]
        col_len = [max(len(row[c]) for row in rows) +
                   2 for c in range(len(rows[0]))]

        with open(output, 'w') as f:
            for row in rows:
                f.write("".join(
                    x + " " * max(1, col_len[i] - len(x)) for i, x in enumerate(row)) + "\n")

    @classmethod
    def read_txt(cls, path: str) -> tuple[torch.Tensor, FeatureMap]:
        with open(path, 'r') as f:
            data = [[float(x) for x in l.split()] for l in f.readlines()]
        tensor = cls.to_tensor(data)
        return tensor, FeatureMap(tensor, {})

    @classmethod
    def read_json(cls, path: str) -> tuple[torch.Tensor, FeatureMap]:
        temporal = False
        with open(path, 'r') as f:
            content = json.load(f)

        if not isinstance(content, dict) or 'data' not in content:
            raise TypeError(
                f'Invalid data. Format sequence under "data" key and provide sequence as a 2D matrix.')

        data = content['data']
        features: dict = content.get('features', dict())

        try:
            data = cls.to_tensor(data)
        except:
            raise ValueError(
                "data must be structured as a 2D matrix, each row with the same number of elements")

        if 'time' in content:
            temporal = True
            time = cls.to_tensor(content['time']).reshape(-1, 1)
            data = torch.cat([time, data], dim=-1)
            features = {str(int(k) + 1): v for (k, v) in features.items()}
            features = {
                **features
            }

        return data, FeatureMap(data, features, temporal)

    @classmethod
    def write_json(cls, tensor: torch.Tensor, path: str, temporal: bool, **kwargs) -> None:
        content = {}
        if temporal:
            content['time'] = tensor[..., 0].tolist()
            content['data'] = tensor[..., 1:].tolist()
        else:
            content['data'] = tensor.tolist()
        features = kwargs.pop('features', {})
        if len(features) > 0:
            content = {'features': features, **content}
        with open(path, 'w') as f:
            json.dump(obj=content, fp=f, indent=4)
