from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Loss(torch.nn.Module, ABC):
    """Base class for all loss functions using ABC."""

    @abstractmethod
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the loss between `y_true` and `y_pred`."""
        pass

    @abstractmethod
    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the residuals between `y_true` and `y_pred`."""
        pass


class MSELoss(Loss):
    """Mean Squared Error loss for regression problems.

    Provides methods to compute the loss and residuals for mean squared error.
    """

    def __init__(self) -> None:
        """Initializes the MeanSquaredError loss function."""
        super().__init__()

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the mean squared error loss.

        Args:
            y_true: Ground truth target tensor.
            y_pred: Predicted tensor.

        Returns:
            A scalar tensor representing the loss.
        """
        return (y_pred - y_true).square().mean()

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the residuals for mean squared error.

        Args:
            y_true: Ground truth target tensor.
            y_pred: Predicted tensor.

        Returns:
            A tensor representing the residuals.
        """
        return y_pred - y_true


class LossWrapper(Loss):
    """Wrapper for a PyTorch loss function to adapt it to the `Loss` interface.

    This class allows wrapping any existing PyTorch loss function to make it compatible
    with the `Loss` interface, providing methods to compute both the loss and residuals.

    The residuals are computed using the square root trick, where the square root of the
    unreduced loss (`reduction='none'`) is used to derive the per-sample residuals.
    This enables compatibility with the Gauss-Newton framework, allowing diverse loss
    functions (e.g., Cross-Entropy, Huber) to be used in least-squares optimization.
    """

    def __init__(self, loss_fn) -> None:
        """Initializes the LossWrapper with a PyTorch loss function.

        Args:
            loss_fn: A callable PyTorch loss function that accepts arguments in the
                format `(input: Tensor, target: Tensor, reduction: str) -> Tensor`.
        """
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the wrapped loss function.

        Args:
            y_true: Ground truth target tensor.
            y_pred: Predicted tensor.

        Returns:
            Tensor: A scalar tensor representing the loss computed by the wrapped loss.
        """
        return self.loss_fn(y_pred, y_true, reduction='mean')

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the residuals using the wrapped loss function.

        Args:
            y_true: Ground truth target tensor.
            y_pred: Predicted tensor.

        Returns:
            Tensor: A tensor representing the residuals, computed as the square root of
            the element-wise loss values without reduction.
        """
        return torch.sqrt(self.loss_fn(y_pred, y_true, reduction='none'))


class L1Loss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.l1_loss)


class HuberLoss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.huber_loss)


class CrossEntropyLoss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.cross_entropy)


class BCELoss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.binary_cross_entropy)


class BCEWithLogitsLoss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.binary_cross_entropy_with_logits)
