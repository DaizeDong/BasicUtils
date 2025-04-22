import base64
import re
import types
import zlib
from argparse import ArgumentTypeError

import numpy as np


def str2none(v: str, extended=True):
    if isinstance(v, types.NoneType):
        return v
    if v.lower() in ("none",) + (("null",) if extended else ()):
        return None
    else:
        raise ValueError(f"Unable to convert \"{v}\" to None.")


def str2bool(v: str, extended=True):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true",) + (("yes", "t", "y", "1") if extended else ()):
        return True
    elif v.lower() in ("false",) + (("no", "f", "n", "0") if extended else ()):
        return False
    else:
        raise ValueError(f"Unable to convert \"{v}\" to bool.")


def _auto_convert_type(v, extended=False):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    try:
        return str2bool(v, extended)
    except ValueError:
        pass
    try:
        return str2none(v, extended)
    except ValueError:
        pass
    return v


def str2dict(v: str, sep=",", extended=False):
    """
    Convert a string to a dictionary.
    The key will be in str type, and the value will be in int/float/bool/None/str type decided by its format.
    Example:
         input: "k1=233,k2=true,k3=hello"
         output: {"k1": 233, "k2": True, "k3": "hello"}
    """
    result = {}
    for item in v.split(sep):
        if "=" not in item:
            raise ArgumentTypeError(f"Invalid format for dictionary item: \"{item}\"")
        key, val = item.split("=", 1)
        result[key.strip()] = _auto_convert_type(val.strip(), extended=extended)
    return result


def str2list(v, sep=",", extended=False):
    """
    Convert a string to a list.
    The value will be in int/float/bool/None/str type decided by its format.
    Example:
         input: "233,true,hello"
         output: [233, True, "hello"]
    """
    result = []
    for item in v.split(sep):
        result.append(_auto_convert_type(item.strip(), extended=extended))
    return result


def str2ndarray(array_str: str, dtype=np.float32) -> np.ndarray:
    """
    Decompress the base64-encoded string back into a numpy array.
    """
    if not array_str.strip():
        return np.array([])
    compressed = base64.b64decode(array_str.encode("utf-8"))
    decompressed = zlib.decompress(compressed)
    return np.frombuffer(decompressed, dtype=dtype)


def ndarray2str(array: np.ndarray, dtype=np.float32, compresslevel: int = 9) -> str:
    """
    Compress a numpy array to a base64-encoded zlib-compressed string.
    """
    if array is None or array.size == 0:
        return ""
    array_bytes = array.astype(dtype).tobytes()
    compressed = zlib.compress(array_bytes, level=compresslevel)
    return base64.b64encode(compressed).decode("utf-8")


def extract_numbers(string, match_float=True, match_sign=True):
    """Extract numbers from a given string."""
    pattern = r"\d+"
    if match_float:
        pattern = r"\d*\.\d+|" + pattern
    if match_sign:
        pattern = r"[-+]?" + pattern
    matches = re.findall(pattern, string)
    numbers = [float(match) if '.' in match else int(match) for match in matches]
    return numbers


def calculate_non_ascii_ratio(string):
    """Calculate the non-ASCII ratio of a given string."""
    if len(string) == 0:
        non_ascii_ratio = 0.0
    else:
        non_ascii_count = sum(1 for char in string if ord(char) >= 128)
        non_ascii_ratio = non_ascii_count / len(string)
    return non_ascii_ratio


def remove_non_ascii_code(string):
    """Use a regular expression to remove all non-ASCII characters"""
    string = re.sub(r'[^\x00-\x7F]+', '', string)
    return string


def replace_non_ascii_code(string):
    """
    Replace common non-ASCII characters with their ASCII counterparts in the given string.

    :param string: Input string with non-ASCII characters.
    :return: String with non-ASCII characters replaced.
    """
    string = re.sub(r'“|”', "\"", string)
    string = re.sub(r'‘|’', "\'", string)
    string = re.sub(r'—|–', "-", string)
    string = re.sub(r'…', "...", string)

    return string
