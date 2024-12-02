from importlib.metadata import metadata

import torch_levenberg_marquardt.damping as damping
import torch_levenberg_marquardt.loss as loss
import torch_levenberg_marquardt.training as training
import torch_levenberg_marquardt.utils as utils

__all__ = ['damping', 'loss', 'training', 'utils']

# Dynamically load metadata from pyproject.toml
meta = metadata('torch-levenberg-marquardt')

__version__ = meta['Version']
__description__ = meta['Summary']
