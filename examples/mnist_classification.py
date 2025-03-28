# %%
# !%reload_ext autoreload
# !%autoreload 2

import time

import torch
import torch_levenberg_marquardt as tlm
from torchmetrics import Accuracy
from torchvision import datasets, transforms

# Set PyTorch to use high precision for matrix multiplication
torch.set_float32_matmul_precision('high')

# Detect CUDA device for acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
# Load MNIST dataset with transformations
transform = transforms.Compose([transforms.ToTensor()])
batch_size = 5000  # Number of samples per batch

# Load training and test datasets
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# Initialize FastDatasetLoader for training and test datasets
train_loader = tlm.utils.FastDataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    repeat=10,
    shuffle=True,
    device=device,
)

test_loader = tlm.utils.FastDataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    repeat=1,
    shuffle=False,
    device=device,
)


# %%
# Define a function to create the convolutional neural network model
def create_conv_model() -> torch.nn.Module:
    """Creates a convolutional neural network for MNIST classification."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0),  # (8, 13, 13)
        torch.nn.ELU(),
        torch.nn.Conv2d(8, 4, kernel_size=4, stride=2, padding=0),  # (4, 5, 5)
        torch.nn.ELU(),
        torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 4, 4)
        torch.nn.ELU(),
        torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 3, 3)
        torch.nn.ELU(),
        torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 2, 2)
        torch.nn.ELU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 2 * 2, 10),  # Fully connected layer for 10 classes
    ).to(device)


# Initialize models for Adam and Levenberg-Marquardt optimization
model = create_conv_model()
model_lm = create_conv_model()

# Print the number of trainable parameters
num_parameters = sum(p.numel() for p in model_lm.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_parameters}')

module = tlm.training.OptimizerModule(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    loss_fn=torch.nn.CrossEntropyLoss(),
)

module_lm = tlm.training.LevenbergMarquardtModule(
    model=model_lm,
    loss_fn=tlm.loss.CrossEntropyLoss(),
    learning_rate=0.05,
    attempts_per_step=10,
    solve_method='cholesky',
)

# %%
# Train the model using the Adam optimizer
print('\nTraining with Adam optimizer...')
t1_start = time.perf_counter()

tlm.utils.fit(
    module,
    train_loader,
    epochs=10,
    metrics={'accuracy': Accuracy(task='multiclass', num_classes=10)},
)

t1_stop = time.perf_counter()
print(f'Training completed. Elapsed time: {t1_stop - t1_start:.2f} seconds')

# %%
# Train the model using the Levenberg-Marquardt algorithm
print('\nTraining with Levenberg-Marquardt...')
t2_start = time.perf_counter()

tlm.utils.fit(
    module_lm,
    train_loader,
    epochs=10,
    metrics={'accuracy': Accuracy(task='multiclass', num_classes=10)},
)

t2_stop = time.perf_counter()
print(f'Training completed. Elapsed time: {t2_stop - t2_start:.2f} seconds')


# %%
# Define evaluation function for the model
def evaluate_model(model, data_loader):
    """Evaluates the model on the provided dataset.

    Args:
        model: The trained model to evaluate.
        data_loader: DataLoader providing the dataset for evaluation.

    Returns:
        tuple: Average loss and accuracy on the dataset.
    """
    model.eval()  # Set model to evaluation mode
    accuracy_metric = Accuracy(task='multiclass', num_classes=10).to(device)
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            # Move data to the appropriate device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)

            # Compute loss for the batch
            loss = torch.nn.functional.cross_entropy(
                y_pred, y_batch, reduction='sum'
            )  # Sum loss over the batch
            total_loss += loss.item()
            total_samples += y_batch.size(0)

            # Update accuracy metric
            accuracy_metric.update(y_pred, y_batch)

    # Compute average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = accuracy_metric.compute().item()

    # Reset metric for future use
    accuracy_metric.reset()
    return avg_loss, accuracy


# %%
# Evaluate both models and print results
adam_loss, adam_acc = evaluate_model(model, test_loader)
lm_loss, lm_acc = evaluate_model(model_lm, test_loader)

print(f'Adam - Test Loss: {adam_loss:.6f}, Test Accuracy: {adam_acc:.2%}')
print(f'Levenberg-Marquardt - Test Loss: {lm_loss:.6f}, Test Accuracy: {lm_acc:.2%}')
