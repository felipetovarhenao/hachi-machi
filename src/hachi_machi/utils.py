import os
import torch
import json
from .features import FeatureMap


def validate_path(file, ext: str | list) -> str:
    if type(ext) == str:
        ext = [ext]
    file = os.path.abspath(os.path.expanduser(file))
    file_ext = os.path.splitext(file)[1]
    if file_ext not in ext:
        raise TypeError(
            f"Invalid extension: {file_ext}. Expected: {', '.join(ext)}")
    return file


def tensor_to_txt(tensor: torch.Tensor, output: str) -> None:
    output = validate_path(output, '.txt')

    rows = [[repr(x) for x in row] for row in tensor.tolist()]
    col_len = [max(len(row[c]) for row in rows) +
               2 for c in range(len(rows[0]))]

    with open(output, 'w') as f:
        for row in rows:
            f.write("".join(
                x + " " * max(1, col_len[i] - len(x)) for i, x in enumerate(row)) + "\n")


def progress(n: int, N: int = 10, size: int = 12):
    nt = min(1, n / N)
    f_size = size * nt
    i_size = int(f_size)
    end_id = int((f_size - i_size) * 7)
    end = ' ⣄⣤⣦⣶⣷⣿'[end_id]
    bar = "⣿" * i_size + end * (end_id > 0)
    return f'{bar}{"⣀" * (size - i_size - (end_id > 0))} {round(nt * 100):.1f}%'


def load_data(file_path: str, device: str = 'cpu') -> tuple[torch.Tensor, dict]:
    temporal = False
    with open(file_path, 'r') as f:
        content = json.load(f)

    if not isinstance(content, dict) or 'data' not in content:
        raise TypeError(
            f'Invalid data. Format sequence under "data" key and provide sequence as a 2D matrix.')

    data = content['data']
    features: dict = content.get('features', dict())

    try:
        data = torch.tensor(data, dtype=torch.float32).to(device)
    except:
        raise ValueError(
            "data must be structured as a 2D matrix, each row with the same number of elements")

    if 'time' in content:
        temporal = True
        time = torch.tensor(
            content['time'], dtype=torch.float32).reshape(-1, 1).to(device)
        data = torch.cat([time, data], dim=-1)
        data[1:, 0] = data[..., 0].diff(dim=-1)
        features = {str(int(k) + 1): v for (k, v) in features.items()}
        features = {
            '0': {
                'type': 'temporal'
            },
            **features
        }

    return data, FeatureMap(data, features, temporal)
