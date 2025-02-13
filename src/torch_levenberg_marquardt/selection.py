from abc import ABC, abstractmethod
from typing import Iterable, Literal

import torch
from torch import Tensor


class ParamSelectionStrategy(ABC):
    """Base class for parameter selection strategies.

    Parameter selection strategies determine which subset of model parameters to update
    during a training step. This could reduce computational, memory requirements and
    act as a regularization for large models.
    """

    @abstractmethod
    def select_parameters(self) -> Tensor:
        """Selects a subset of parameters for the current training step.

        Returns:
            The indices of the selected parameters in a flattened parameter vector.
        """
        pass


class RandomSelectionStrategy(ParamSelectionStrategy):
    """Randomly selects a fixed-size subset of parameters at each step.

    This strategy picks a random subset of the model's parameters from the
    flattened parameter vector. It is often used to limit computations and
    memory usage, making it more feasible to work with very large models.

    Args:
        params: An iterable of model parameters.
        subset_size: The number of parameters to select at each step.

    Raises:
        ValueError: If `subset_size` exceeds the total number of parameters.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        subset_size: int,
    ) -> None:
        self.params = list(params)
        self.total_num_params = sum(p.numel() for p in self.params)

        if subset_size > self.total_num_params:
            raise ValueError(
                f'subset_size ({subset_size}) cannot exceed the total number of '
                f'parameters ({self.total_num_params}).'
            )

        self.subset_size = subset_size
        self.device = self.params[0].device if self.params else torch.device('cpu')

    def select_parameters(self) -> Tensor:
        """Selects a random subset of parameters.

        Returns:
            The indices of the selected parameters in a flattened parameter vector.
        """
        param_indices = torch.randperm(self.total_num_params, device=self.device)
        selected = param_indices[: self.subset_size]
        return selected.sort().values


class LayerSelectionStrategy(ParamSelectionStrategy):
    """Selects parameters corresponding to a single parameter group each step.

    This strategy returns a contiguous block of parameter indices for exactly one
    parameter tensor at a time (e.g., corresponding to a single layer). The parameter
    selection can be performed randomly or in a round-robin fashion (cyclic mode).

    Args:
        params: An iterable of model parameters.
        mode: The selection mode. Should be either 'random' or 'cyclic'.

    Raises:
        ValueError: If `mode` is not 'random' or 'cyclic'.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        mode: Literal['random', 'cyclic'] = 'random',
    ) -> None:
        if mode not in ('random', 'cyclic'):
            raise ValueError("mode must be either 'random' or 'cyclic'.")

        self.params = list(params)
        self.mode = mode
        self.device = self.params[0].device if self.params else torch.device('cpu')

        # Precompute the slices corresponding to each parameter
        current_index = 0
        self.param_slices: list[tuple[int, int]] = []
        for p in self.params:
            size = p.numel()
            self.param_slices.append((current_index, current_index + size))
            current_index += size

        # For 'cyclic' mode, keep track of the current parameter index
        self.current_param_idx = 0

    def select_parameters(self) -> Tensor:
        """Selects parameters corresponding to a single parameter tensor.

        Returns:
            The indices of the selected parameters in a flattened parameter vector.
        """
        if not self.params:
            return torch.tensor([], dtype=torch.long, device=self.device)

        if self.mode == 'random':
            param_idx = int(
                torch.randint(low=0, high=len(self.params), size=(1,)).item()
            )
        else:
            param_idx = self.current_param_idx
            self.current_param_idx = (self.current_param_idx + 1) % len(self.params)

        start, end = self.param_slices[param_idx]
        return torch.arange(start, end, device=self.device, dtype=torch.long)
