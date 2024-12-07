from typing import Generator

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
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
    overwrite_progress_bar: bool = True,
    update_every_n_steps: int = 1,
) -> None:
    """Fit function with support for TrainingModule and torchmetrics.

    Trains the model for a specified number of epochs. It supports logging metrics using
    `torchmetrics` and provides detailed progress tracking using `tqdm`.

    Args:
        training_module: A `TrainingModule` encapsulating the training logic.
        dataloader: A PyTorch DataLoader.
        epochs: The number of epochs.
        metrics: Optional dict of torchmetrics.Metric objects.
        overwrite_progress_bar: If True, mimic a single-line progress bar similar
            to PyTorch Lightning (old bars overwritten).
        update_every_n_steps: Update the progress bar and displayed logs every n steps.
    """
    assert update_every_n_steps > 0
    device = training_module.device
    steps = len(dataloader)
    stop_training = False

    if metrics:
        metrics = {name: metric.to(device) for name, metric in metrics.items()}

    for epoch in range(epochs):
        if stop_training:
            break

        # Create a new progress bar for this epoch
        progress_bar = tqdm(
            total=steps,
            desc=f'Epoch {epoch + 1}/{epochs}',
            leave=not overwrite_progress_bar,  # Leave bar if overwrite is False
            dynamic_ncols=True,
        )
        total_loss = 0.0
        steps_since_update = 0

        for step, (inputs, targets) in enumerate(dataloader):
            # Ensure that inputs and targets are on the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)

            # Perform a training step
            outputs, loss, stop_training, logs = training_module.training_step(
                inputs, targets
            )

            total_loss += loss.item()

            # Update metrics if provided
            if metrics:
                for name, metric in metrics.items():
                    metric(outputs, targets)

            # Format logs
            formatted_logs = {'loss': f'{loss:.4e}'}
            if metrics:
                for name, metric in metrics.items():
                    formatted_logs[name] = metric.compute().item()
            for key, value in logs.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                formatted_logs[key] = (
                    f'{value:.4e}' if isinstance(value, float) else str(value)
                )

            steps_since_update += 1
            if (
                steps_since_update == update_every_n_steps
                or step == steps - 1
                or stop_training
            ):
                # Update the progress bar and logs
                progress_bar.update(steps_since_update)
                progress_bar.set_postfix(formatted_logs)
                steps_since_update = 0

            if stop_training:
                # End early, ensure progress bar remains visible
                progress_bar.leave = True
                break

        # Reset metrics at the end of the epoch
        if metrics:
            for metric in metrics.values():
                metric.reset()

        # Epoch summary
        avg_loss = total_loss / steps
        if overwrite_progress_bar:
            progress_bar.set_postfix({'epoch_avg_loss': f'{avg_loss:.4e}'})
        else:
            progress_bar.write(
                f'Epoch {epoch + 1} complete. Average loss: {avg_loss:.4e}'
            )

        # Ensure the final progress bar is left visible
        if epoch == epochs - 1 or stop_training:
            progress_bar.leave = True

        progress_bar.close()

    # Final training summary
    if overwrite_progress_bar:
        print(f'Training complete. Final epoch average loss: {avg_loss:.4e}')


class FastDataLoader(DataLoader):
    """A lightweight and efficient data loader optimized for small datasets.

    This class addresses the performance bottleneck caused by the overhead of
    `torch.utils.data.DataLoader` when dealing with small models or datasets that can
    fit entirely into RAM or GPU memory. By preloading the entire dataset into memory,
    it eliminates unnecessary overhead and significantly accelerates the training.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        repeat: int = 1,
        shuffle: bool = False,
        device: torch.device | str = 'cpu',
    ) -> None:
        """
        Initializes the FastDataLoader.

        Args:
            dataset: A PyTorch Dataset from which data and targets will be extracted.
            batch_size: Number of samples per batch.
            repeat: Number of times to repeat the dataset.
            shuffle: If True, shuffle the data at the start of each repetition.

        Raises:
            ValueError: If data and targets have mismatched sizes.
        """
        super().__init__(dataset=dataset, batch_size=batch_size)

        self.repeat = repeat
        self.shuffle = shuffle

        data, targets = zip(*((data, target) for data, target in dataset))

        assert len(data) > 0
        assert len(targets) > 0

        self.data = (
            torch.stack(data)
            if isinstance(data[0], torch.Tensor)
            else torch.tensor(data)
        ).to(device)
        self.targets = (
            torch.stack(targets)
            if isinstance(targets[0], torch.Tensor)
            else torch.tensor(targets)
        ).to(device)

        if self.data.size(0) != self.targets.size(0):
            raise ValueError("Data and targets must have the same number of samples.")

        self.n_samples = self.data.size(0)

    def __iter__(self) -> Generator[tuple[Tensor, Tensor], None, None]:
        """
        Generates batches of data and targets.

        Yields:
            A tuple (batch_data, batch_targets) for each batch.
        """
        assert self.batch_size

        for _ in range(self.repeat):
            indices = torch.arange(self.n_samples)
            if self.shuffle:
                indices = indices[torch.randperm(self.n_samples)]

            for i in range(0, self.n_samples, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                yield self.data[batch_indices], self.targets[batch_indices]

    def __len__(self) -> int:
        """
        Returns the total number of batches.

        Returns:
            Total number of batches across all repeats.
        """
        assert self.n_samples
        assert self.batch_size

        batches_per_epoch = (self.n_samples + self.batch_size - 1) // self.batch_size
        return self.repeat * batches_per_epoch
