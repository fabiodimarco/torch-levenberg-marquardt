# PyTorch Levenberg-Marquardt

[![PyPI](https://img.shields.io/pypi/v/torch-levenberg-marquardt)](https://pypi.org/project/torch-levenberg-marquardt/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabiodimarco/torch-levenberg-marquardt/blob/main/examples/torch_levenberg_marquardt.ipynb)

A PyTorch implementation of the **Levenberg-Marquardt (LM)** optimization algorithm, supporting **mini-batch training** for both **regression** and **classification** problems. It leverages GPU acceleration and offers an extensible framework, supporting diverse loss functions and customizable damping strategies.

A TensorFlow implementation is also available: [tf-levenberg-marquardt](https://github.com/fabiodimarco/tf-levenberg-marquardt)

For more information on the theory behind the Levenberg-Marquardt and Gauss-Newton algorithms, refer to the following resources:
[Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm),
[Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm).

## Why Levenberg-Marquardt?

First-order methods like SGD and Adam dominate large-scale neural network training due to their efficiency and scalability, making them the only viable option for models with millions or billions of parameters.
However, for smaller models, second-order methods can offer faster convergence and sometimes succeed where first-order methods fail.

The **Levenberg-Marquardt algorithm** strikes a balance:
- It builds on the Gauss-Newton method, using second-order information for faster convergence.
- Adaptive damping enhances stability, mitigating issues that can arise in the standard Gauss-Newton algorithm.

This makes it a strong choice for problems with manageable model sizes.

## Features

- **Versatile Loss Support**: Leverage the square root trick to apply LM with any PyTorch-supported loss function.
- **Mini-batch Training**: Scale LM to large datasets for both regression and classification tasks.
- **Custom Damping Strategies**: Adapt the damping factor dynamically for stable optimization.
- **Split Jacobian Matrix Computation**: Split the Computation of the Jacobian and Hessian matrix approximation to reduce memory usage.
- **Custom Param Selection Strategies**: Select a subset of model parameters to update during the training step.

## Supported Loss Functions

The following loss functions are supported out of the box:
- `MSELoss`
- `L1Loss`
- `HuberLoss`
- `CrossEntropyLoss`
- `BCELoss`
- `BCEWithLogitsLoss`

Additional loss functions can be added by implementing custom residual definitions.

## Implemented damping strategies

* **Standard**: $\large J^T J + \lambda I$
* **Fletcher**: $\large J^T J + \lambda ~ \text{diag}\(J^T J\hspace{0.1em})$
* **Custom**: Support for defining custom damping strategies.

## Installation

To install the library, use pip:
```bash
pip install torch-levenberg-marquardt
```

## Development Setup

To contribute or modify the code, clone the repository and install it in editable mode:

#### CUDA-enabled systems
```bash
conda env create -f environment_cuda.yml
```

#### CPU-only systems
```bash
conda env create -f environment_cpu.yml
```

#### MacOS systems
```bash
conda env create -f environment_macos.yml
```

## Usage Examples

### Training Loop

The `utils.fit` function provides an example of how to implement a PyTorch training loop using the `training.LevenbergMarquardtModule`.

```python
import torch_levenberg_marquardt as tlm

# The fit function provides an example of how to train your model in PyTorch training loop
tlm.utils.fit(
  tlm.training.LevenbergMarquardtModule(
    model=model,
    loss_fn=tlm.loss.MSELoss(),
    learning_rate=1.0,
    attempts_per_step=10,
    solve_method='qr',
  ),
  train_loader,
  epochs=50,
)
```

### PyTorch Lightning Training

The class `utils.CustomLightningModule` provides an example of how to implement a PyTorch Lightning module that uses the `training.LevenbergMarquardtModule`:

```python
import torch_levenberg_marquardt as tlm
from pytorch_lightning import Trainer

# Wrap your model with the Levenberg-Marquardt training module
lm_module = tlm.utils.CustomLightningModule(
  tlm.training.LevenbergMarquardtModule(
    model=model,
    loss_fn=tlm.loss.MSELoss(),
    learning_rate=1.0,
    attempts_per_step=10,
    solve_method='qr',
  )
)

# Train using PyTorch Lightning
trainer = Trainer(max_epochs=50, accelerator='gpu', devices=1)
trainer.fit(lm_module, train_loader)
```

## Adapting Levenberg-Marquardt for Any Loss Function

The Levenberg-Marquardt algorithm is designed to solve **least-squares problems** or the form:

$$
\large \underset{W}{\text{argmin}} \enspace S\left(W\right) = \sum_{i=1}^N r_i\left(W\right)^2, \quad r_i = y_i - f\left(x_i, W\right)
$$

This might seem to restrict it to **Mean Squared Error loss**. However, it is possible to use the **square root trick** to adapt LM to any loss that is guaranteed to always be positive. Suppose we have a loss function of the form:

$$
\large \ell\left(y, f\left(x, W\right)\right) ≥ 0 \quad \text{for all valid inputs and outputs}
$$

the residuals $\large r_i$ in LM can be redefined to generalize the algorithm as:

$$
\large r_i = \sqrt{\ell\left(y, f\left(x, W\right)\right)}
$$

### Examples:

#### 1. Mean Squared Error Loss

$$
\large r_i = \sqrt{\left(y_i - f\left(x_i, W\right)\right)^2} \quad \Rightarrow \quad r_i = \left|\hspace{0.3em}y_i - f\left(x_i, W\right)\right| \quad \Rightarrow \quad r_i = y_i - f\left(x_i, W\right)
$$

```python
class MSELoss(Loss):
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return (y_pred - y_true).square().mean()

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return y_pred - y_true
```

#### 2. Cross-Entropy Loss

$$
\large r_i = \sqrt{-y_i \log\left(f\left(x_i, W\right)\right)}
$$

```python
class CrossEntropyLoss(Loss):
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(y_pred, y_true, reduction='mean')

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return torch.sqrt(torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none'))
```

## Levenberg-Marquardt Algorithm Explanation

### Gauss-Newton Hessian Matrix Approximation: Derivation

The **Gauss-Newton method** provides an efficient way to optimize least-squares problems by approximating the second-order derivatives of the objective function:

$$
\large \underset{W}{\text{argmin}} \enspace S\left(W\right) = \sum_{i=1}^N r_i\left(W\right)^2, \quad r_i = r\left(y_i, f\left(x_i, W\right)\right)
$$

where:

- $\large r_i$ are the residuals derived from a general loss function,
- $\large f\left(x_i, W\right)$ represents the model output for input $\large x_i$ and parameters $\large W$,
- $\large N$ is the number of data points.

In what follows, the Gauss-Newton algorithm will be derived from Newton's method for function optimization via an approximation.
The recurrence relation for Newton's method for minimizing a function $\large S$ of parameters $\large W$ is:

$$
\large W^{\hspace{0.3em} s+1} = W^{\hspace{0.3em} s} - H^{\hspace{0.1em}-1} g
$$

where $\large g$ denotes the gradient vector of $\large S$, and $\large H$ denotes the Hessian matrix of $\large S$.

The gradient of $\large S$ with respect to $\large W$ is given by:

$$
\large g_j = 2 \sum_{i=1}^N r_i \frac{\partial r_i}{\partial w_j},
$$

Elements of the Hessian are calculated by differentiating the gradient elements, $\large g_j$, with respect to $\large w_k$:

$$
\large H_{jk} = 2 \sum_{i=1}^N \left( \frac{\partial r_i}{\partial w_j} \frac{\partial r_i}{\partial w_k} + r_i \frac{\partial^2 r_i}{\partial w_j \partial w_k} \right)
$$

The Gauss-Newton method is obtained by ignoring the second-order derivative terms (the second term in this expression). That is, the Hessian is approximated by

$$
\large H_{jk} \approx 2 \sum_{i=1}^N J_{ij} J_{ik}, \quad J_{ij} = \frac{\partial r_i}{\partial w_j}
$$


where $\large J_{ij} = \partial r_i / \partial w_j$ are entries of the Jacobian matrix $\large J_r$. Note that when the exact Hessian is evaluated near an exact fit we have near-zero
$\large r_i$, so the second term becomes near-zero as well, which justifies the approximation. The gradient and the approximate Hessian can be written in matrix notation as

$$
\large g = 2 J_r^T r, \quad H \approx 2 J_r^T J_r
$$

### Levenberg-Marquardt: Damped Update

While the Gauss-Newton method is powerful, its instability near singularity in $\large J_r^T J_r$ is a limitation. The **Levenberg-Marquardt algorithm** mitigates this by introducing a damping factor $\large \lambda$:

$$
\large H = J_r^T J_r + \lambda I
$$

This ensures numerical stability when $\large J_r^T J_r$ is ill-conditioned. The update rule becomes:

$$
\large W^{\hspace{0.3em} s+1} = W^{\hspace{0.3em} s} - (J_r^T J_r + \lambda I)^{-1} J_r^T r
$$

The damping $\large \lambda$ is changed based on an adaptive strategy. During each training step, the LM algorithm attempts to find an update for the model that reduces the loss. In each attempt, new model parameters are computed, and the resulting loss is compared to the previous loss. If `new_loss < loss`, the new parameters are accepted. Otherwise, the old parameters are restored, and a new attempt is made with an adjusted damping factor.

## Memory, Speed, and Convergence Considerations

To achieve optimal performance from the training algorithm, it is important to carefully choose the batch size and the number of model parameters.

### Jacobian Matrix Dimensions

The LM algorithm minimizes the least-squares objective:

$$
\large \underset{W}{\text{argmin}} \sum_{i=1}^N \left[r\left(y_i, f\left(x_i, W\right)\right)\right]^2
$$

The Jacobian matrix $\large J_r$ captures the partial derivatives of the residuals with respect to the model parameters:

$$
\large J_r[i, j] = \frac{\partial r_i}{\partial w_j}
$$

For a batch of size $\large B$, the Jacobian matrix $\large J_r$ has dimensions:

$$
\large J_r \in \mathbb{R}^{N \times P}
$$

where:
- $\large W$: Model parameters (weights),
- $\large N$: Total number of residuals, determined by $\large N = B \cdot O$,
- $\large B$: Batch size,
- $\large O$: Number of outputs per sample,
- $\large r\left(y_i, f\left(x_i, W\right)\right)$: Residuals, which are computed differently depending on the chosen loss function.
- $\large P$: Total number of parameters in the model.

### Two Formulations of the Update Rule

The LM update is chosen based on whether the system is  **overdetermined** $\large (N > P)$ or **underdetermined** $\large (N < P)$, with the algorithm checking the system type for each batch.

#### 1. Overdetermined Systems $\large (N > P)$

Update formula:

$$
\large W^{\hspace{0.3em} s+1} = W^{\hspace{0.3em} s} - (J_r^T J_r + \lambda I)^{-1} J_r^T r
$$

The Size of the matrix to invert $\large J_r^T J_r + \lambda I$ is $\large [P \times P]$.

#### 2. Underdetermined Systems $\large (N < P)$

Update formula:

$$
\large W^{\hspace{0.3em} s+1} = W^{\hspace{0.3em} s} - J_r^T (J_r J_r^T + \lambda I)^{-1} r
$$

The Size of the matrix to invert $\large J_r J_r^T + \lambda I$ is $\large [N \times N]$.

### Split Jacobian Matrix Computation (Overdetermined Case Only)

The memory required to store $\large J_r$ scales with both $\large N$ and $\large P$. Thus, directly storing and computing $\large J_r$ is often infeasible.
However, for overdetermined systems $(N > P)$, the properties of $\large J_r^T J_r$ can be leveraged to compute it incrementally through **Jacobian splitting**.
This functionality can be controlled in the code by setting the `jacobian_max_num_rows` argument.

Rather than constructing the full $\large J_r$ at once, divide the computation into smaller sub-batches.

#### 1. Batch Splitting:

Split the full batch into $\large M$ sub-batches of size $\large B_s$:

$$
\large M = \lceil B / B_s \rceil,
$$

where $\large B$ is the full batch size and $\large B_s$ is the sub-batch size.

#### 2. Sub-Batch Computation

For each sub-batch $\large i$, compute the corresponding residuals $\large \bar{r}_i$ of size $\large N_s$ and Jacobians $\large \bar{J}_i$ of size $\large [N_s \times P]$.

Where $\large N_s$ the number of sub-residuals, is determined by $\large B_s \cdot O$.

#### 3. Incremental Accumulation
Instead of storing the entire $\large J_r$, compute $\large J_r^T J_r$ and $\large J_r^T r$ incrementally:

$$
\large J_r^T J_r = \sum_{i=1}^M \bar{J}_i^T \bar{J}_i,
$$

$$
J_r^T r = \sum_{i=1}^M \bar{J}_i^T \bar{r}_i
$$


#### Memory Upper Bound

When using the Split Jacobian computation, the memory usage is primarily determined by the size of $\large J_r^T J_r$, which is $\large [P \times P]$, where $\large P$ is the number of model parameters. While this approach reduces overall memory requirements compared to storing the full Jacobian matrix $\large J_r$, the size of $\large J_r^T J_r$ can still become a bottleneck for models with a very large number of parameters, potentially leading to out-of-memory (OOM) errors.

## Results
### Curve fitting
A simple curve-fitting example is implemented in `examples/sinc_curve_fitting.py` and `examples/sinc_curve_fitting_lightning.py`. The function `y = sinc(10 * x)` is fitted using a shallow neural network with 61 parameters.
Despite the simplicity of the problem, first-order methods such as Adam fail to converge, whereas Levenberg-Marquardt converges rapidly with very low loss values. The learning rate values were chosen experimentally based on the results obtained by each algorithm.

Here the results with Adam for 10000 epochs and learning_rate=0.01
```
Training with Adam optimizer...
Epoch 9999: 100%|██████████| 20/20 [00:00<00:00, 81.07it/s, loss_step=0.000461, loss_epoch=0.000412]
`Trainer.fit` stopped: `max_epochs=10000` reached.
Epoch 9999: 100%|██████████| 20/20 [00:00<00:00, 80.21it/s, loss_step=0.000461, loss_epoch=0.000412]
Training completed. Elapsed time: 2604.56 seconds
```
Here the results with Levenberg-Marquardt for 100 epochs and learning_rate=1.0
```
Training with Levenberg-Marquardt...
Epoch 49: 100%|██████████| 20/20 [00:00<00:00, 64.07it/s, loss_step=2.79e-7, damping_factor=1e-6, attempts=3.000, loss_epoch=3.31e-7]
`Trainer.fit` stopped: `max_epochs=50` reached.
Epoch 49: 100%|██████████| 20/20 [00:00<00:00, 63.35it/s, loss_step=2.79e-7, damping_factor=1e-6, attempts=3.000, loss_epoch=3.31e-7]
Training completed. Elapsed time: 16.60 seconds
```
#### Plot
<img width="700" alt="Curve Fitting Comparison Plot" src="https://github.com/user-attachments/assets/f01eee5a-4e7f-417c-99de-f269771a6c22">

### Mnist dataset classification
A common MNIST classification example is implemented in `examples/mnist_classification.py.py` and `examples/mnist_classification_lightning.py.py`. The classification is performed using a convolutional neural network with 1026 parameters.
Both optimization methods achieve roughly the same accuracy on the training and test sets; however, Levenberg-Marquardt requires significantly fewer epochs, automatically stopping the training at epoch 8.

Here the results with Adam for 100 epochs and learning_rate=0.01
```
Training with Adam optimizer...
Training with Adam optimizer...
Epoch 99: 100%|██████████| 12/12 [00:01<00:00,  8.90it/s, accuracy=0.970, loss_step=0.0977, loss_epoch=0.0986]
`Trainer.fit` stopped: `max_epochs=100` reached.
Epoch 99: 100%|██████████| 12/12 [00:01<00:00,  8.88it/s, accuracy=0.970, loss_step=0.0977, loss_epoch=0.0986]
Training completed. Elapsed time: 125.33 seconds

Adam - Test Loss: 0.089224, Test Accuracy: 97.32%
```
Here the results with Levenberg-Marquardt for 10 epochs and learning_rate=0.05
```
Train using Levenberg-Marquardt
Training with Levenberg-Marquardt...
Epoch 8:  83%|████████▎ | 10/12 [00:02<00:00,  4.64it/s, accuracy=0.977, loss_step=0.0683, damping_factor=1e+10, attempts=3.000, loss_epoch=0.0742]
Training completed. Elapsed time: 22.10 seconds

Levenberg-Marquardt - Test Loss: 0.076580, Test Accuracy: 97.59%
```

## Requirements

 * python>3.9
 * torch>=2.0.0
 * numpy>=1.22
 * pytorch-lightning>=1.9
 * tqdm
 * torchmetrics>=0.11.0
