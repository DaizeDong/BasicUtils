import numpy as np
import torch


def convert_to_type(value, value_type):
    assert value_type in ("default", "numpy", "torch")

    if value_type == "default" and type(value) not in (int, float):
        if type(value) is np.ndarray:
            return value.item()
        elif type(value) is torch.Tensor:
            return value.item()
        else:
            raise TypeError(f"{type(value)} is not a valid type!")
    elif value_type == "numpy" and type(value) is not np.ndarray:
        if type(value) in (int, float):
            return np.array((value,))
        elif type(value) is torch.Tensor:
            return value.cpu().numpy()
        else:
            raise TypeError(f"{type(value)} is not a valid type!")
    elif value_type == "torch" and type(value) is not torch.Tensor:
        if type(value) in (int, float):
            return torch.tensor((value,))
        elif type(value) is np.ndarray:
            return torch.from_numpy(value)
        else:
            raise TypeError(f"{type(value)} is not a valid type!")
    else:
        if type(value) in (int, float):
            return value
        elif type(value) is np.ndarray:
            return value.reshape(1)
        elif type(value) is torch.Tensor:
            return value.detach().reshape(1)


class NumberCache:
    def __init__(self, value_type="default"):
        available_value_type = ("default", "numpy", "torch")
        assert value_type in available_value_type

        self.value_type = value_type
        self.cache = {}
        self.cnt = {}

    def add(self, name, value):
        if name not in self.cache:
            self.cache[name] = 0
        if name not in self.cnt:
            self.cnt[name] = 0
        value = convert_to_type(value, self.value_type)
        self.cache[name] += value
        self.cnt[name] += 1

    def get(self, name, clear_cache=True):
        return self.get_sum(name, clear_cache=clear_cache)

    def get_sum(self, name, clear_cache=True):
        value = self.cache[name]
        if clear_cache:
            self.cache[name] = 0
            self.cnt[name] = 0
        return value

    def get_mean(self, name, clear_cache=True):
        value = self.cache[name] / self.cnt[name]
        if clear_cache:
            self.cache[name] = 0
            self.cnt[name] = 0
        return value


class ListCache:
    def __init__(self, value_type="default"):
        available_value_type = ("default", "numpy", "torch")
        assert value_type in available_value_type

        self.value_type = value_type
        self.cache = {}

    def add(self, name, value):
        if name not in self.cache:
            self.cache[name] = []
        value = convert_to_type(value, self.value_type)
        self.cache[name].append(value)

    def get_sum(self, name, clear_cache=True):
        if self.value_type == "default":
            value = sum(self.cache[name])
        elif self.value_type == "torch":
            value = torch.sum(torch.cat(self.cache[name], dim=0))
        elif self.value_type == "numpy":
            value = np.sum(np.concatenate(self.cache[name], axis=0))

        if clear_cache:
            self.cache[name] = []

        return value

    def get_mean(self, name, clear_cache=True):
        if self.value_type == "default":
            value = sum(self.cache[name]) / len(self.cache[name])
        elif self.value_type == "torch":
            value = torch.mean(torch.cat(self.cache[name], dim=0).float())
        elif self.value_type == "numpy":
            value = np.mean(np.concatenate(self.cache[name], axis=0))

        if clear_cache:
            self.cache[name] = []

        return value
