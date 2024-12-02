import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import Tensor

# pyright: reportPrivateImportUsage=false
from torch.func import functional_call, jacrev

from .damping import DampingStrategy, StandardDampingStrategy
from .loss import Loss, MSELoss

logger = logging.getLogger(__name__)


class TrainingModule(ABC):
    """Abstract base class defining a training interface."""

    @abstractmethod
    def training_step(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, bool, dict[str, Any]]:
        """Performs a single training step."""
        pass

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        """Returns the model being trained."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Returns the device of the model's parameters."""
        pass


class LevenbergMarquardtModule(TrainingModule):
    """Levenberg-Marquardt training module."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Loss | None = None,
        damping_strategy: DampingStrategy | None = None,
        jacobian_max_num_rows: int | None = None,
        learning_rate: float = 1.0,
        attempts_per_step: int = 10,
        solve_method: Literal['qr', 'cholesky', 'solve'] = 'qr',
    ) -> None:
        """Initializes `LevenbergMarquardtModule` instance.

        Args:
            model: The model to be trained, expected to inherit from `torch.nn.Module`.
            loss_fn: A custom loss function inheriting from `Loss`.
                Defaults to `MSELoss()`.
            damping_strategy: Damping strategy to use during training.
                Defaults to `StandardDampingStrategy`.

            jacobian_max_num_rows: If set, and the number of residuals exceeds the
                number of variables (overdetermined case), the Jacobian is computed in
                smaller input batches and then accumulated to form the full Gauss-Newton
                Hessian matrix approximation. This approach reduces memory usage and
                improves computational efficiency. Each Jacobian matrix will contain at
                most `jacobian_max_num_rows` rows.
            learning_rate: Specifies the step size for updating the model parameters.
                The update is performed using the formula `w = w - lr * updates`,
                where `updates` are calculated by the Levenberg-Marquardt algorithm.
            attempts_per_step: Defines the maximum number of attempts allowed during a
                training step to compute a valid model update that reduces the loss on
                the current batch. During each attempt, new model parameters are
                computed, and the resulting loss (`new_loss`) is compared to the
                previous loss. If `new_loss < loss`, the new parameters are accepted.
                Otherwise, the old parameters are restored, and a new attempt is made
                with an adjusted damping factor. If the maximum number of attempts is
                reached without reducing the loss, the step is finalized with the last
                computed parameters, even if they do not decrease the loss.
            solve_method: Solver to use for the linear system. Options:
                - 'qr': QR decomposition (robust, slower).
                - 'cholesky': Cholesky decomposition (fast, less stable).
                - 'solve': Direct solve (balanced speed and robustness).
        """
        self._model = model

        self.loss_fn = loss_fn or MSELoss()
        self.damping_strategy = damping_strategy or StandardDampingStrategy()
        self.jacobian_max_num_rows = jacobian_max_num_rows
        self.learning_rate = learning_rate
        self.attempts_per_step = attempts_per_step

        self.solve_method = solve_method

        # Initialize damping factor
        self.damping_factor = self.damping_strategy.get_starting_value()

        # Backup for model parameters
        self._backup_parameters: list[Tensor] = []

        # Parameters required for computing updates
        self._params = [p for p in self._model.parameters() if p.requires_grad]
        self._splits = [p.numel() for p in self._params]
        self._shapes = [p.shape for p in self._params]
        self._num_parameters = sum(self._splits)
        self._num_outputs = None

    def reset(self) -> None:
        """Resets the trainer state.

        Resets the damping factor and internal counters.
        """
        self._num_outputs = None
        self.damping_factor = self.damping_strategy.get_starting_value()

    def backup_parameters(self) -> None:
        """Backs up the current model parameters."""
        self._backup_parameters = [param.clone() for param in self._params]

    def restore_parameters(self) -> None:
        """Restores the model parameters from the backup."""
        with torch.no_grad():
            for param, backup in zip(self._params, self._backup_parameters):
                param.copy_(backup)

    def _compute_num_outputs(self, inputs: Tensor, targets: Tensor) -> int:
        """Computes the number of outputs from the model.

        Args:
            inputs: Input tensor.
            targets: Target tensor.

        Returns:
            The number of outputs produced by the model.
        """
        # Create dummy inputs and targets with batch size zero
        input_shape = inputs.shape[1:]  # Exclude batch dimension
        target_shape = targets.shape[1:]

        dummy_inputs = torch.zeros(
            (1,) + input_shape, dtype=inputs.dtype, device=inputs.device
        )
        dummy_targets = torch.zeros(
            (1,) + target_shape, dtype=targets.dtype, device=targets.device
        )

        # Pass inputs through the model
        outputs = self._model(dummy_inputs)

        # Compute residuals
        residuals = self.loss_fn.residuals(dummy_targets, outputs)

        return residuals.numel()

    def _solve(self, matrix: Tensor, rhs: Tensor) -> Tensor:
        """Solves the linear system using QR decomposition.

        Args:
            matrix: The matrix representing the linear system.
            rhs: The right-hand side vector.

        Returns:
            The solution vector.
        """

        if self.solve_method == 'qr':
            q, r = torch.linalg.qr(matrix)
            y = torch.matmul(q.transpose(-2, -1), rhs)
            return torch.linalg.solve_triangular(r, y, upper=True)
        elif self.solve_method == 'cholesky':
            L = torch.linalg.cholesky(matrix)
            y = torch.linalg.solve_triangular(L, rhs, upper=False)
            return torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        elif self.solve_method == 'solve':
            return torch.linalg.solve(matrix, rhs)
        else:
            raise ValueError(
                f"Invalid solve_method '{self.solve_method}'. "
                "Choose from 'qr', 'cholesky', 'solve'."
            )

    def _compute_jacobian(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the Jacobian of the residuals with respect to model parameters.

        This method uses `torch.func.jacrev` to compute the Jacobian efficiently.

        Args:
            inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            targets: Target tensor of shape `(batch_size, target_dim, ...)`.

        Returns:
            tuple: A tuple containing:
                - jacobian: The Jacobian matrix of shape `(num_residuals, num_params)`.
                - residuals: Residual vector of shape `(num_residuals, 1)`.
                - outputs: Model outputs of shape `(batch_size, target_dim, ...)`.
        """
        # Filter parameters that require gradients
        params = {
            name: p for name, p in self._model.named_parameters() if p.requires_grad
        }
        buffers = dict(self._model.named_buffers())

        # Capture outputs and residuals
        captured_outputs = None
        captured_residuals = None

        # Define a function that computes residuals given parameters
        def compute_residuals(params):
            nonlocal captured_outputs, captured_residuals
            outputs = functional_call(self._model, {**params, **buffers}, inputs)
            residuals = self.loss_fn.residuals(targets, outputs).view(-1)
            captured_outputs = outputs
            captured_residuals = residuals
            return residuals

        # Compute the Jacobian of residuals with respect to parameters
        jacobian_params: dict[str, Tensor]
        jacobian_params = jacrev(compute_residuals)(params)  # type: ignore

        # Retrieve captured outputs and residuals
        assert captured_outputs is not None
        assert captured_residuals is not None
        outputs = captured_outputs
        residuals = captured_residuals

        # Flatten and concatenate the gradients to form the Jacobian matrix
        jacobian_matrices = []
        for _, j in jacobian_params.items():
            # j has shape: (num_residuals, param_shape...)
            # Reshape to (num_residuals, param_numel)
            j = j.reshape(residuals.numel(), -1)
            jacobian_matrices.append(j)

        # Shape: (num_residuals, total_num_params)
        jacobian = torch.cat(jacobian_matrices, dim=1)

        residuals = residuals.unsqueeze(-1)

        return jacobian, residuals, outputs

    def _sliced_gauss_newton(
        self, inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gauss-Newton approximation for overdetermined systems using slicing.

        This method handles large overdetermined systems by dividing the input into
        smaller slices. For each slice, the Jacobian matrix is computed and used to
        incrementally build the full Gauss-Newton Hessian approximation and the
        right-hand side (RHS) vector. This approach reduces memory usage while
        maintaining computational accuracy.

        Args:
            inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            targets: Target tensor of shape `(batch_size, output_dim, ...)`.

        Returns:
            tuple:
                - JJ: The accumulated Gauss-Newton Hessian approximation of shape
                    `(num_parameters, num_parameters)`.
                - rhs: The right-hand side vector of shape `(num_parameters, 1)`.
                - outputs: The concatenated model outputs of shape
                    `(batch_size, output_dim, ...)`.
        """
        assert self.jacobian_max_num_rows is not None
        assert self._num_outputs is not None

        slice_size = self.jacobian_max_num_rows // self._num_outputs
        batch_size = inputs.shape[0]
        num_slices = batch_size // slice_size
        remainder = batch_size % slice_size

        JJ = torch.zeros(
            (self._num_parameters, self._num_parameters),
            dtype=inputs.dtype,
            device=inputs.device,
        )

        rhs = torch.zeros(
            (self._num_parameters, 1),
            dtype=inputs.dtype,
            device=inputs.device,
        )

        outputs_list = []

        for i in range(num_slices):
            idx_start = i * slice_size
            idx_end = (i + 1) * slice_size
            _inputs = inputs[idx_start:idx_end]
            _targets = targets[idx_start:idx_end]

            J, residuals, _outputs = self._compute_jacobian(_inputs, _targets)
            outputs_list.append(_outputs)

            JJ += J.t().matmul(J)  # JJ = JJ + J' * J
            rhs += J.t().matmul(residuals)  # rhs = rhs + J' * residuals

        if remainder > 0:
            _inputs = inputs[num_slices * slice_size :]
            _targets = targets[num_slices * slice_size :]

            J, residuals, _outputs = self._compute_jacobian(_inputs, _targets)
            outputs_list.append(_outputs)

            JJ += J.t().matmul(J)  # JJ = JJ + J' * J
            rhs += J.t().matmul(residuals)  # rhs = rhs + J' * residuals

        outputs = torch.cat(outputs_list, dim=0)

        return JJ, rhs, outputs

    def training_step(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, bool, dict[str, Any]]:
        """Performs a single training step.

        Args:
            inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            targets: Target tensor of shape `(batch_size, target_dim, ...)`.

        Returns:
            tuple: A tuple containing:
                - outputs: Model outputs for the given inputs.
                - loss: The computed loss value.
                - stop_training: Whether training should stop.
                - logs: Additional metadata (e.g., damping factor, attempts).
        """
        if self._num_outputs is None:
            # Initialize during the first train step
            self._num_outputs = self._compute_num_outputs(inputs, targets)
            self.backup_parameters()

        batch_size = inputs.shape[0]
        num_residuals = batch_size * self._num_outputs
        overdetermined = num_residuals >= self._num_parameters

        if overdetermined:
            if self.jacobian_max_num_rows:
                # overdetermined reduced memory sliced JJ computation
                JJ, rhs, outputs = self._sliced_gauss_newton(inputs, targets)
            else:
                # overdetermined
                J, residuals, outputs = self._compute_jacobian(inputs, targets)
                JJ = J.t().matmul(J)  # JJ = J' * J
                rhs = J.t().matmul(residuals)  # rhs = J' * residuals
                J = None
        else:
            # underdetermined
            J, residuals, outputs = self._compute_jacobian(inputs, targets)
            JJ = J.matmul(J.t())  # JJ = J * J'
            rhs = residuals  # rhs = residuals

        # Normalize for numerical stability
        normalization_factor = 1.0 / batch_size
        JJ *= normalization_factor
        rhs *= normalization_factor

        # Compute the current loss value
        loss = self.loss_fn(targets, outputs)

        stop_training = False
        attempt = 0
        damping_factor = self.damping_strategy.init_step(self.damping_factor, loss)

        while True:  # Infinite loop, break conditions inside
            params_updated = False

            # Try to update the parameters
            try:
                # Apply damping to the Gauss-Newton Hessian approximation
                JJ_damped = self.damping_strategy.apply(damping_factor, JJ)

                # Compute the updates:
                # - Overdetermined: updates = (J' * J + damping)^-1 * J'*residuals
                # - Underdetermined: updates = J' * (J * J' + damping)^-1 * residuals
                updates = self._solve(JJ_damped, rhs)

                if not overdetermined:
                    assert J is not None
                    updates = J.t().matmul(updates)

                # Check if updates are finite
                if torch.all(torch.isfinite(updates)):
                    params_updated = True
                    # Split and reshape the updates
                    updates = torch.split(updates.view(-1), self._splits)
                    updates = [
                        update.view(shape)
                        for update, shape in zip(updates, self._shapes)
                    ]

                    # Apply the updates to the model parameters
                    with torch.no_grad():
                        for param, update in zip(self._params, updates):
                            param.add_(-self.learning_rate * update)

            except Exception as e:
                logger.warning(f'An exception occurred: {e}')

            if attempt < self.attempts_per_step:
                attempt += 1

                if params_updated:
                    # Compute the new loss value
                    outputs = self._model(inputs)
                    new_loss = self.loss_fn(targets, outputs)

                    if new_loss < loss:
                        # Accept the new model parameters and backup them
                        loss = new_loss
                        damping_factor = self.damping_strategy.decrease(
                            damping_factor, loss
                        )
                        self.backup_parameters()
                        break

                    # Restore the old parameters and try a new damping factor
                    self.restore_parameters()

                # Adjust the damping factor for the next attempt
                damping_factor = self.damping_strategy.increase(damping_factor, loss)

                # Check if training should stop
                stop_training = self.damping_strategy.stop_training(
                    damping_factor, loss
                )
                if stop_training:
                    break
            else:
                break

        # Update the damping factor for the next train_step
        self.damping_factor = damping_factor

        logs = {
            'damping_factor': damping_factor,
            'attempts': attempt,
        }

        return outputs, loss, stop_training, logs

    @property
    def model(self) -> torch.nn.Module:
        """The model being trained.

        Returns:
            torch.nn.Module: The neural network model used for training.
        """
        return self._model

    @property
    def device(self) -> torch.device:
        """The device on which the model's parameters are stored.

        This determines whether the model is on a CPU, GPU, or another device.

        Returns:
            torch.device: The device of the model's parameters.
        """
        return next(self._model.parameters()).device


class OptimizerModule(TrainingModule):
    """Train step for standard optimizers (e.g., Adam, SGD).

    This module provides a simple training loop for models using standard
    first-order optimization methods like SGD or Adam. It wraps the model,
    optimizer, and loss function, and provides a `training_step` method
    to perform parameter updates.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ) -> None:
        """Initializes the OptimizerModule.

        Args:
            model: The neural network model to be trained.
            optimizer: The optimizer used for training (e.g., SGD, Adam).
            loss_fn: The loss function used to compute the training objective.
        """
        self._model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def training_step(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, bool, dict[str, Any]]:
        """Performs a training step using a standard optimizer.

        This method computes the loss for the given inputs and targets, performs
        backpropagation, and updates the model parameters using the optimizer.

        Args:
            inputs: Input tensor for the model, with shape depending on the task.
            targets: Target tensor, with shape depending on the task.

        Returns:
            tuple:
                - outputs: The model's predictions for the given inputs.
                - loss: The computed loss value.
                - stop_training: Always False, as it does not handle early stopping.
                - logs: An empty dictionary, as it does not provide additional logging.
        """
        # Forward pass
        outputs = self._model(inputs)
        loss: Tensor = self.loss_fn(outputs, targets)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return outputs, loss, False, {}

    @property
    def model(self) -> torch.nn.Module:
        """The model being trained.

        Returns:
            torch.nn.Module: The neural network model used for training.
        """
        return self._model

    @property
    def device(self) -> torch.device:
        """The device on which the model's parameters are stored.

        This determines whether the model is on a CPU, GPU, or another device.

        Returns:
            torch.device: The device of the model's parameters.
        """
        return next(self._model.parameters()).device
