from typing import Any

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


def tree_unsqueeze(tree: Any, dim: int = 0) -> Any:
    """Recursively unsqueeze every tensor in a pytree along the given dimension.

    Args:
        tree: The pytree containing tensors and non-tensor leaves.
        dim: The dimension along which to unsqueeze each tensor.

    Returns:
        A new pytree with every tensor unsqueezed along the specified dimension.
    """
    return tree_map(lambda x: x.unsqueeze(dim) if isinstance(x, Tensor) else x, tree)


def tree_to_device(tree: Any, device: torch.device | str) -> Any:
    """Recursively move all tensor leaves in a pytree to the specified device.

    Args:
        tree: The pytree containing tensors and non-tensor leaves.
        device: The target device (e.g. CPU or GPU) to move the tensors to.

    Returns:
        A new pytree with every tensor moved to the specified device.
    """
    return tree_map(lambda x: x.to(device) if isinstance(x, Tensor) else x, tree)


def tree_first_tensor(tree: Any) -> Tensor:
    """Return the first tensor found in the pytree.

    Args:
        tree: The pytree to search for a tensor.

    Returns:
        The first tensor encountered in the pytree.

    Raises:
        ValueError: If no tensor is found in the pytree.
    """
    flat_leaves, _ = tree_flatten(tree)
    first_tensor = next((x for x in flat_leaves if isinstance(x, Tensor)), None)
    if first_tensor is None:
        raise ValueError('No tensor found in the given pytree.')
    return first_tensor


def tree_cat(trees: list[Any], dim: int = 0) -> Any:
    """Concatenate a list of pytrees along the given dimension.

    Tensor leaves are concatenated using torch.cat; for non-tensor leaves only the
    first one is taken.

    Args:
        trees: A list of pytrees to concatenate.
        dim: The dimension along which to concatenate the tensor leaves.

    Returns:
        A new pytree with tensor leaves concatenated along the specified dimension.
    """
    # Flatten the first tree to get its structure.
    _, tree_def = tree_flatten(trees[0])
    # Flatten each tree to get the leaves, all trees must share the same structure.
    all_leaves = [tree_flatten(tree)[0] for tree in trees]
    cat_leaves = []
    for group in zip(*all_leaves):
        if isinstance(group[0], Tensor):
            cat_leaves.append(torch.cat(group, dim=dim))
        else:
            cat_leaves.append(group[0])
    return tree_unflatten(cat_leaves, tree_def)


def tree_slice(tree: Any, start: int, end: int) -> Any:
    """Slice every indexable leaf in the pytree from start to end.

    For tensors and sequence types (e.g., lists, tuples), standard slicing is applied.

    Args:
        tree: The pytree whose indexable leaves will be sliced.
        start: The start index of the slice.
        end: The end index of the slice.

    Returns:
        A new pytree with each indexable leaf sliced from start to end.
    """
    return tree_map(
        lambda x: (
            x[start:end] if hasattr(x, '__getitem__') and not isinstance(x, str) else x
        ),
        tree,
    )


def tree_indices(tree: Any, indices: list[int]) -> Any:
    """Select elements from every indexable leaf in the pytree using the provided list
    of indices.

    For any indexable object, this function first attempts to index directly with the
    list of indices. If that fails, it falls back to iterating over the indices
    and reconstructing the object.

    Args:
        tree: The pytree whose indexable leaves will be indexed.
        indices: A list of indices to select from each indexable leaf.

    Returns:
        A new pytree with each indexable leaf selected by the given indices.
    """
    return tree_map(
        lambda x: (
            x[indices]
            if hasattr(x, '__getitem__') and not isinstance(x, str)
            else type(x)(x[i] for i in indices)
        ),
        tree,
    )
