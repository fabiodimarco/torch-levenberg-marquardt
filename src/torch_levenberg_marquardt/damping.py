from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor


class DampingStrategy(ABC):
    """Base class for damping strategies in Levenberg-Marquardt optimization."""

    @abstractmethod
    def reset(self) -> None:
        """Resets any state to its initial value."""
        pass

    @abstractmethod
    def get_current_damping(self) -> Tensor:
        """Retrieves the current damping factor."""
        pass

    @abstractmethod
    def initialize_step(self, loss: Tensor) -> None:
        """Initializes any state before a training step."""
        pass

    @abstractmethod
    def on_successful_update(self, loss: Tensor) -> None:
        """Adjust the damping factor after a successful update."""
        pass

    @abstractmethod
    def on_unsuccessful_update(self, loss: Tensor) -> None:
        """Adjust the damping factor after an unsuccessful update."""
        pass

    @abstractmethod
    def stop_attempts(self, loss: Tensor) -> bool:
        """Determines if the update should be accepted with no further attempts"""
        pass

    @abstractmethod
    def stop_training(self, loss: Tensor) -> bool:
        """Checks if training should stop based on the damping factor."""
        pass

    @abstractmethod
    def apply(self, JJ: Tensor) -> Tensor:
        """Applies damping to the Gauss-Newton Hessian approximation."""
        pass


class StandardDampingStrategy(DampingStrategy):
    """Standard Levenberg-Marquardt damping strategy.

    This is used inside the Trainer as a generic class. Many damping strategies can be
    implemented using the same interface.
    """

    def __init__(
        self,
        starting_value: float = 1e-3,
        dec_factor: float = 0.1,
        inc_factor: float = 10.0,
        min_value: float = 1e-10,
        max_value: float = 1e10,
        damping_mode: Literal['standard', 'adaptive', 'fletcher'] = 'standard',
        conditional_stopping: bool = True,
        auto_reset: bool = False,
    ) -> None:
        """Initializes `StandardDampingStrategy` instance.

        Args:
            starting_value: Used to initialize the Trainer internal damping_factor.
            dec_factor: Used in the train_step to decrease the damping_factor when
                new_loss < loss.
            inc_factor: Used in the train_step to increase the damping_factor when
                new_loss >= loss.
            min_value: Used as a lower bound for the damping_factor. Higher values
                improve numerical stability in the resolution of the linear system, at
                the cost of slower convergence.
            max_value: Used as an upper bound for the damping_factor, and as a condition
                to stop the training process.
            damping_mode: Specifies the damping mode. Options are:
                - 'standard': Standard damping using the identity matrix (default).
                - 'adaptive': Apply adaptive scaling with max(diagonal(JJ)).
                - 'fletcher': Use Fletcher's modification for damping.
            conditional_stopping: If True, stops training based on damping conditions.
            auto_reset: If True, resets the damping factor when `stop_attempts` is True.
        """
        self.starting_value = torch.tensor(starting_value)
        self.dec_factor = torch.tensor(dec_factor)
        self.inc_factor = torch.tensor(inc_factor)
        self.min_value = torch.tensor(min_value)
        self.max_value = torch.tensor(max_value)
        self.damping_mode = damping_mode
        self.conditional_stopping = conditional_stopping
        self.auto_reset = auto_reset

        self.damping_factor = torch.tensor(starting_value)

    def reset(self) -> None:
        """Resets the damping factor to the starting value."""
        self.damping_factor = self.starting_value

    def get_current_damping(self) -> Tensor:
        """Retrieves the current damping factor."""
        return self.damping_factor

    def initialize_step(self, loss: Tensor) -> None:
        """Initializes any state before a training step."""
        pass

    def on_successful_update(self, loss: Tensor) -> None:
        """Decreases the damping factor.

        Args:
            loss: The current loss value.

        Returns:
            The decreased damping factor.
        """
        self.damping_factor = torch.max(
            self.damping_factor * self.dec_factor, self.min_value
        )

    def on_unsuccessful_update(self, loss: Tensor) -> None:
        """Increases the damping factor.

        Args:
            loss: The current loss value.

        Returns:
            The increased damping factor.
        """
        self.damping_factor = torch.min(
            self.damping_factor * self.inc_factor, self.max_value
        )

    def stop_attempts(self, loss: Tensor) -> bool:
        """Determines if further attempts should be stopped and performs auto-reset."""
        should_stop = bool((self.damping_factor >= self.max_value).item())
        if self.auto_reset and should_stop:
            self.reset()
        return should_stop

    def stop_training(self, loss: Tensor) -> bool:
        """Determines whether to stop training based on the damping factor.

        Args:
            loss: The current loss value.

        Returns:
            True if the damping factor exceeds the maximum value, False otherwise.
        """
        return self.stop_attempts(loss) and self.conditional_stopping

    def apply(self, JJ: Tensor) -> Tensor:
        """Applies the damping to the Gauss-Newton Hessian approximation.

        Args:
            JJ: The Gauss-Newton Hessian approximation matrix.

        Returns:
            The damped Hessian matrix.
        """
        if self.damping_mode == 'standard':
            damping_matrix = torch.eye(JJ.shape[0], dtype=JJ.dtype, device=JJ.device)
        elif self.damping_mode == 'adaptive':
            damping_matrix = torch.eye(JJ.shape[0], dtype=JJ.dtype, device=JJ.device)
            damping_matrix = damping_matrix * torch.max(torch.abs(torch.diagonal(JJ)))
        elif self.damping_mode == 'fletcher':
            damping_matrix = torch.diag(torch.diagonal(JJ))
        else:
            raise ValueError(
                f"Invalid damping_mode '{self.damping_mode}'. Expected one of "
                f"'standard', 'adaptive', or 'fletcher'."
            )

        return JJ + self.damping_factor * damping_matrix
