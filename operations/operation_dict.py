from typing import Any, Dict, List, Tuple, Union


def dict_to_args_str(args_dict: Dict[str, Any]) -> str:
    shell_args = []

    for key, value in args_dict.items():
        arg_name = "--" + key.replace("_", "-")

        if value is None:
            pass
        elif isinstance(value, bool) and value:
            shell_args.append(arg_name)
        elif isinstance(value, (int, float, str)):
            shell_args.append(f"{arg_name}")
            shell_args.append(f"{value}")
        elif isinstance(value, list):  # cat with ","
            shell_args.append(f"{arg_name}")
            shell_args.append(",".join(map(str, value)))
        elif isinstance(value, dict):  # cat with ","
            shell_args.append(f"{arg_name}")
            shell_args.append(",".join(f"{k}={v}" for k, v in value.items()))
        else:
            raise TypeError(f"Unsupported type for argument '{key}': {type(value)}")

    return " ".join(shell_args)


def reverse_dict(input_dict, aggregate_same_results=True):
    output_dict = {}
    for key, value in input_dict.items():
        if value not in output_dict:
            output_dict[value] = key
        else:
            if aggregate_same_results:
                if not isinstance(output_dict[value], List):
                    output_dict[value] = [output_dict[value]]
                output_dict[value].append(key)
            else:
                raise ValueError("Input dictionary does not satisfy the one-to-one mapping condition.")
    return output_dict


def safe_pop(input_dict: Dict[str, Any], key: Union[str, List[str], Tuple[str]]) -> Any:
    """Safely pop keys from a dictionary, returning None if the key does not exist."""
    if isinstance(key, (list, tuple)):
        return [safe_pop(input_dict, k) for k in key]
    else:
        return input_dict.pop(key) if key in input_dict else None
