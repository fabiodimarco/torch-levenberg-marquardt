from abc import ABC, abstractmethod

import torch
from torch import Tensor


class DampingStrategy(ABC):
    """Base class for damping strategies in Levenberg-Marquardt optimization."""

    @abstractmethod
    def get_starting_value(self) -> Tensor:
        """Returns the initial damping factor."""
        pass

    @abstractmethod
    def init_step(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        """Initializes the damping factor for a new training step."""
        pass

    @abstractmethod
    def decrease(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        """Decreases the damping factor after a successful update."""
        pass

    @abstractmethod
    def increase(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        """Increases the damping factor after an unsuccessful update."""
        pass

    @abstractmethod
    def stop_training(self, damping_factor: Tensor, loss: Tensor) -> bool:
        """Checks if training should stop based on the damping factor."""
        pass

    @abstractmethod
    def apply(self, damping_factor: Tensor, JJ: Tensor) -> Tensor:
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
        adaptive_scaling: bool = False,
        fletcher: bool = False,
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
            adaptive_scaling: Scales the damping_factor adaptively multiplying it
                with max(diagonal(JJ)).
            fletcher: Replaces the identity matrix with the diagonal of the
                Gauss-Newton Hessian approximation, so that there is larger movement
                along the directions where the gradient is smaller. This avoids slow
                convergence in the direction of small gradient.
        """
        self.starting_value = torch.tensor(starting_value)
        self.dec_factor = torch.tensor(dec_factor)
        self.inc_factor = torch.tensor(inc_factor)
        self.min_value = torch.tensor(min_value)
        self.max_value = torch.tensor(max_value)
        self.adaptive_scaling = adaptive_scaling
        self.fletcher = fletcher

    def get_starting_value(self) -> Tensor:
        """Gets the initial damping factor.

        Returns:
            Tensor: A scalar tensor representing the initial damping factor.
        """
        return self.starting_value

    def init_step(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        """Initializes the damping factor for a new training step.

        Args:
            damping_factor: The current damping factor.
            loss: The current loss value.

        Returns:
            Tensor: The initialized damping factor, identical to the input by default.
        """
        return damping_factor

    def decrease(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        """Decreases the damping factor.

        Args:
            damping_factor: The current damping factor.
            loss: The current loss value.

        Returns:
            The decreased damping factor.
        """
        return torch.max(damping_factor * self.dec_factor, self.min_value)

    def increase(self, damping_factor: Tensor, loss: Tensor) -> Tensor:
        """Increases the damping factor.

        Args:
            damping_factor: The current damping factor.
            loss: The current loss value.

        Returns:
            The increased damping factor.
        """
        return torch.min(damping_factor * self.inc_factor, self.max_value)

    def stop_training(self, damping_factor: Tensor, loss: Tensor) -> bool:
        """Determines whether to stop training based on the damping factor.

        Args:
            damping_factor: The current damping factor.
            loss: The current loss value.

        Returns:
            True if the damping factor exceeds the maximum value, False otherwise.
        """
        return bool((damping_factor >= self.max_value).item())

    def apply(self, damping_factor: Tensor, JJ: Tensor) -> Tensor:
        """Applies the damping to the Gauss-Newton Hessian approximation.

        Args:
            damping_factor: The current damping factor.
            JJ: The Gauss-Newton Hessian approximation matrix.

        Returns:
            The damped Hessian matrix.
        """
        if self.fletcher:
            damping_matrix = torch.diag(torch.diagonal(JJ))
        else:
            damping_matrix = torch.eye(JJ.shape[0], dtype=JJ.dtype, device=JJ.device)

        scaler = torch.tensor(1.0, dtype=JJ.dtype, device=JJ.device)
        if self.adaptive_scaling:
            scaler = torch.max(torch.abs(torch.diagonal(JJ)))

        damping_matrix = scaler * damping_factor * damping_matrix
        return JJ + damping_matrix
