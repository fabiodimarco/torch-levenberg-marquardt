import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

from .training import TrainingModule


class CustomLightningModule(pl.LightningModule):
    """PyTorch Lightning Module with support for custom TrainingModule and torchmetrics.

    This module integrates a `TrainingModule` for custom training logic and allows
    logging metrics using `torchmetrics`. It disables PyTorch Lightning's automatic
    optimization, enabling full control over the training loop.
    """

    def __init__(
        self,
        training_module: TrainingModule,
        metrics: dict[str, Metric] | None = None,
    ) -> None:
        """Initializes the CustomLightningModule.

        Args:
            training_module: An instance of `TrainingModule` encapsulating custom
                training logic.
            metrics: A dictionary of `torchmetrics.Metric` objects for evaluation,
                with metric names as keys. Defaults to an empty dictionary.
        """
        super().__init__()
        self.training_module = training_module
        self.metrics = metrics or {}
        self.automatic_optimization = False  # Disable automatic optimization

    def on_fit_start(self) -> None:
        """Moves metrics to the device where the model resides."""
        device = self.device  # Get the device of the model
        for metric in self.metrics.values():
            metric.to(device)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Performs a single training step.

        Args:
            batch: A tuple `(inputs, targets)` containing the input and target tensors.
            batch_idx: The index of the batch (required by PyTorch Lightning).

        Returns:
            Tensor: The computed loss for the current batch.
        """
        inputs, targets = batch

        # Perform training step using the custom TrainingModule
        outputs, loss, stop_training, logs = self.training_module.training_step(
            inputs, targets
        )

        # Convert logs into tensors for compatibility
        logs = {
            key: (
                value
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, dtype=torch.float32)
            )
            for key, value in logs.items()
        }

        # Compute metrics if defined
        metric_logs = {}
        for name, metric in self.metrics.items():
            metric_logs[name] = metric(outputs, targets)

        # Log metrics
        self.log_dict(
            {name: value.item() for name, value in metric_logs.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        # Log loss and additional logs
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(logs, on_step=True, on_epoch=False, prog_bar=True)

        # Signal Lightning to stop training if necessary
        self.trainer.should_stop = stop_training

        return loss

    def configure_optimizers(self) -> list:
        """Prevents PyTorch Lightning from performing optimizer steps.

        Returns:
            list: An empty list as no optimizers are used in this module.
        """
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the model.

        Args:
            x: Input tensor to the model.

        Returns:
            Tensor: The output of the model.
        """
        return self.training_module.model(x)


def fit(
    training_module: TrainingModule,
    dataloader: DataLoader,
    epochs: int,
    metrics: dict[str, Metric] | None = None,
) -> None:
    """Fit function with support for TrainingModule and individual torchmetrics.

    Manually trains the model for a specified number of epochs. This function
    supports logging metrics using `torchmetrics` and provides detailed progress
    tracking using `tqdm`.

    Args:
        training_module: A `TrainingModule` encapsulating the training logic.
        dataloader: A PyTorch DataLoader providing `(inputs, targets)` tuples.
        epochs: The number of epochs to train the model.
        metrics: A dictionary of `torchmetrics.Metric` objects for evaluation,
            with metric names as keys. Defaults to None.
    """
    steps = len(dataloader)
    stop_training = False

    # Move metrics to the correct device
    device = training_module.device
    if metrics:
        metrics = {name: metric.to(device) for name, metric in metrics.items()}

    for epoch in range(epochs):
        if stop_training:
            break

        print(f'Epoch {epoch + 1}/{epochs}')

        progress_bar = tqdm(
            enumerate(dataloader),
            total=steps,
            desc=f'Epoch {epoch + 1}/{epochs}',
            leave=True,
            dynamic_ncols=True,
        )

        total_loss = 0

        for _, (inputs, targets) in progress_bar:
            # Move inputs and targets to the correct device
            inputs, targets = inputs.to(device), targets.to(device)

            if stop_training:
                break

            # Perform a training step
            outputs, loss, stop_training, logs = training_module.training_step(
                inputs, targets
            )

            # Compute metrics if defined
            metric_logs = {}
            if metrics:
                for name, metric in metrics.items():
                    metric_logs[name] = metric(outputs, targets)

            total_loss += loss.item()

            # Format logs for the progress bar
            formatted_logs = {'loss': f'{loss:.4e}'}
            formatted_logs.update(
                {name: value.item() for name, value in metric_logs.items()}
            )
            for key, value in logs.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                formatted_logs[key] = (
                    f'{value:.4e}' if isinstance(value, float) else str(value)
                )

            # Update the progress bar with formatted metrics
            progress_bar.set_postfix(formatted_logs)

        # Reset all metrics at the end of each epoch
        if metrics:
            for metric in metrics.values():
                metric.reset()

        print(f'Epoch {epoch + 1} complete. Average loss: {total_loss / steps:.4e}')

    print('Training complete.')
