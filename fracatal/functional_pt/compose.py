import numpy as np
import numpy.random as npr
import torch

from fracatal.functional_pt.convolve import ft_convolve
from fracatal.functional_pt.pad import pad_2d

def make_kernel_field(kernel_radius, dim=126, default_dtype=torch.float32):

  x =  torch.arange(-dim / 2, dim / 2 + 1, 1)
  xx, yy = torch.meshgrid(x,x)

  rr = torch.sqrt(xx**2 + yy**2) / kernel_radius
  rr = torch.tensor(rr, dtype=default_dtype)

  return rr
