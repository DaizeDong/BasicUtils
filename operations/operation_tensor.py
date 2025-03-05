from typing import Dict, List, Sized, Tuple, Union

import torch


def move_tensors_to_device(input, device, strict=False):
    """Recursively move tensors in input to the specified device."""
    if isinstance(input, torch.Tensor):
        input.to(device)

    elif isinstance(input, Dict):
        for key, value in input.items():
            input[key] = move_tensors_to_device(value, device, strict)

    elif isinstance(input, List):
        for i, value in enumerate(input):
            input[i] = move_tensors_to_device(value, device, strict)

    else:
        if strict:
            raise TypeError('Unsupported input type:', type(input))

    return input


def tensors_are_same(input1, input2, strict=False):
    """Recursively check if two tensor structures are the same."""
    if type(input1) != type(input2):
        return False

    elif isinstance(input1, torch.Tensor):
        return torch.equal(input1, input2)

    elif isinstance(input1, Dict):
        if set(input1.keys()) != set(input2.keys()):
            return False
        return all(tensors_are_same(input1[key], input2[key]) for key in input1)

    elif isinstance(input1, List):
        if len(input1) != len(input2):
            return False
        return all(tensors_are_same(item1, item2) for item1, item2 in zip(input1, input2))

    else:
        if strict:
            raise TypeError('Unsupported input type:', type(input1))
        return False


def tensor2numbers(input):
    if input is None:
        return input

    elif isinstance(input, Dict):
        for key, value in input.items():
            if isinstance(value, torch.Tensor):
                input[key] = value.tolist()
        return input

    elif isinstance(input, List):
        for i in range(len(input)):
            if isinstance(input[i], torch.Tensor):
                input[i] = input[i].tolist()
        return input

    elif isinstance(input, torch.Tensor):
        return input.tolist()

    else:
        raise TypeError(input)


def last_true_position(mask):
    """Return the index of the last true value in each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    true_mask_cnt = torch.sum(mask, dim=1).unsqueeze(1)
    last_true_mask = (mask.cumsum(dim=1) == true_mask_cnt) & mask
    last_true_position = last_true_mask.nonzero()[:, 1].unsqueeze(1)
    return last_true_position


def turn_last_true_mask_to_false(mask, true_mask_cnt=None):
    """Turn the last true value to false for each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    if true_mask_cnt is None:
        true_mask_cnt = torch.sum(mask, dim=1).unsqueeze(1)
    turn_position_indices = (mask.cumsum(dim=1) == true_mask_cnt)
    converted_mask = mask.clone()
    converted_mask[turn_position_indices] = False
    return converted_mask


def turn_first_true_mask_to_false(mask):
    """Turn the first true value to false for each row in a mask matrix."""
    # mask: shape(batch_size, seq_len)
    turn_position_indices = (mask.cumsum(dim=1) == 1)
    converted_mask = mask.clone()
    converted_mask[turn_position_indices] = False
    return converted_mask


def equalize_true_in_mask(mask, max_true_threshold):
    """
    Adjust the mask such that each row has the same number of True values,
    specified by max_true_threshold, by turning the last few True values to False if a row exceeds the threshold.
    """
    # mask: shape(batch_size, seq_len)
    retain_true_positions = (mask.cumsum(dim=1) <= max_true_threshold)
    mask = (mask & retain_true_positions)
    return mask


def pass_kernel_function(tensor, criterion, allow_nan=False):
    if criterion == "plain":
        return tensor
    elif criterion == "sqrt":
        if not allow_nan and torch.any(tensor < 0):
            raise ValueError("Detected negative value in the tensor! This will cause the result to be \"nan\"!")
        return torch.sqrt(tensor)
    elif criterion == "l1":
        return torch.abs(tensor)
    elif criterion == "l2":
        return tensor * tensor
    else:
        raise NotImplementedError


def concat_tensors(input: Union[List, Tuple], dim=0, auto_reshape=False, strict=False):
    """
    Recursively concatenate a list of nested tensors along the specified dimension
    at the outermost level while preserving the structure.

    Args:
        input (list/tuple): A list/tuple where each element has the same nested structure (dict/list/tensor).
        dim (int): The dimension along which to concatenate tensors.
        auto_reshape (bool): If True, automatically reshapes tensors to have the same dimension.
        strict (bool): If True, raises an error for unsupported input types. Otherwise, combines them into a list.

    Returns:
        The concatenated structure with tensors merged at the outermost level.
        If non-tensor elements exist, they are grouped into lists instead of being concatenated.

    Example:
        >>> data = [
        ...     {
        ...      "a": torch.randn(2, 3),
        ...      "b": [{"x": torch.randn(2, 3), "y": "text1"},
        ...            {"x": torch.randn(2, 3), "y": 123}]
        ...     },
        ...     {
        ...      "a": torch.randn(2, 3),
        ...      "b": [{"x": torch.randn(5, 3), "y": "text2"},
        ...            {"x": torch.randn(5, 3), "y": 456}]
        ...     }
        ... ]
        >>> result = concat_tensors(data, dim=0)
        >>> print(result)
        {
            "a": tensor of shape (4, 3),
            "b": [
                {"x": tensor of shape (7, 3), "y": ["text1", "text2"]},
                {"x": tensor of shape (7, 3), "y": [123, 456]}
            ]
        }
    """
    if len(input) == 0:
        return None

    first_elem = input[0]

    if isinstance(first_elem, torch.Tensor):
        if auto_reshape and first_elem.ndim < dim + 1:  # the dimension is not enough, add with 1
            cat_shape = first_elem.shape + (1,) * (dim + 1 - first_elem.ndim)
            return torch.cat([t.reshape(cat_shape) for t in input], dim=dim)
        else:
            return torch.cat(input, dim=dim)

    elif isinstance(first_elem, Dict):
        return {
            key: concat_tensors([item[key] for item in input], dim=dim, auto_reshape=auto_reshape, strict=strict)
            for key in first_elem
        }

    elif isinstance(first_elem, Sized):
        return [
            concat_tensors([item[i] for item in input], dim=dim, auto_reshape=auto_reshape, strict=strict)
            for i in range(len(first_elem))
        ]

    else:
        if strict:
            raise TypeError(f"Unsupported element type: {type(first_elem)}")
        return [item for item in input]
