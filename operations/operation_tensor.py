import torch


def move_tensors_to_device(input, device, strict=False):
    """Recursively move tensors in input to the specified device."""
    if isinstance(input, torch.Tensor):
        input.to(device)

    elif isinstance(input, dict):
        for key, value in input.items():
            input[key] = move_tensors_to_device(value, device, strict)

    elif isinstance(input, list):
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

    elif isinstance(input1, dict):
        if set(input1.keys()) != set(input2.keys()):
            return False
        return all(tensors_are_same(input1[key], input2[key]) for key in input1)

    elif isinstance(input1, list):
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

    elif isinstance(input, dict):
        for key, value in input.items():
            if isinstance(value, torch.Tensor):
                input[key] = value.tolist()
        return input

    elif isinstance(input, list):
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
