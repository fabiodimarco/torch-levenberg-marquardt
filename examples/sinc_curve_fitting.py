# %%
# !%reload_ext autoreload
# !%autoreload 2

import time

import torch
import torch_levenberg_marquardt as tlm
from bokeh.plotting import figure, output_notebook, show
from torch.utils.data import TensorDataset

# Set PyTorch to use high precision for matrix multiplication
torch.set_float32_matmul_precision('high')

# Detect CUDA device for acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
# Generate synthetic dataset for training
input_size = 20000  # Total number of data points
batch_size = 1000  # Number of samples per batch

# Generate training inputs and outputs (y = sinc(10 * x))
x_train = torch.linspace(-1, 1, input_size, dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.sinc(10 * x_train).to(device)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train, y_train)
train_loader = tlm.utils.FastDataLoader(
    train_dataset,
    batch_size=batch_size,
    repeat=10,
    shuffle=True,
    device=device,
)


# %%
# Define a function to create the neural network model
def create_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 1),
    ).to(device)


# Initialize models for Adam and Levenberg-Marquardt optimization
model = create_model()
model_lm = create_model()

# Print the number of trainable parameters
num_parameters = sum(p.numel() for p in model_lm.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_parameters}')

module = tlm.training.OptimizerModule(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    loss_fn=torch.nn.MSELoss(),
)

module_lm = tlm.training.LevenbergMarquardtModule(
    model=model_lm,
    loss_fn=tlm.loss.MSELoss(),
    learning_rate=1.0,
    attempts_per_step=10,
    solve_method='qr',
)

# %%
# Train the model using the Adam optimizer
print('Training with Adam optimizer...')
t1_start = time.perf_counter()

tlm.utils.fit(
    module,
    train_loader,
    epochs=10,
)

t1_stop = time.perf_counter()
print(f'Training completed. Elapsed time: {t1_stop - t1_start:.2f} seconds')

# %%
# Train the model using the Levenberg-Marquardt algorithm
print('Training with Levenberg-Marquardt...')
t2_start = time.perf_counter()

tlm.utils.fit(
    module_lm,
    train_loader,
    epochs=10,
)

t2_stop = time.perf_counter()
print(f'Training completed. Elapsed time: {t2_stop - t2_start:.2f} seconds')

# %%
# Evaluate both models on the training set and plot the results
print('Generating predictions and plotting results...')

# Generate predictions for the entire training dataset
with torch.no_grad():
    y_pred_adam = model(x_train)
    y_pred_lm = model_lm(x_train)

# Activate notebook output for Bokeh plots
output_notebook()

# Flatten tensors for plotting
x_train_np = x_train.cpu().numpy().flatten()
y_train_np = y_train.cpu().numpy().flatten()
y_pred_adam_np = y_pred_adam.cpu().numpy().flatten()
y_pred_lm_np = y_pred_lm.cpu().numpy().flatten()

# Create a Bokeh figure
p = figure(
    title='Comparison of Optimization Methods',
    x_axis_label='x_train',
    y_axis_label='y_values',
    width=800,
    height=400,
)

# Add reference and predictions to the plot
p.line(x_train_np, y_train_np, line_width=2, color='blue', legend_label='Reference')
p.line(
    x_train_np,
    y_pred_adam_np,
    line_width=2,
    line_dash='dashed',
    color='green',
    legend_label='Adam',
)
p.line(
    x_train_np,
    y_pred_lm_np,
    line_width=2,
    line_dash='dashed',
    color='red',
    legend_label='Levenberg-Marquardt',
)

# Customize legend
p.legend.title = 'Methods'
p.legend.label_text_font_size = '10pt'
p.legend.location = 'top_left'

# Display the plot in the notebook
show(p)
