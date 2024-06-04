import numpy as np

import torch

from torch.fft import fft2, ifft2, fftshift, ifftshift

from fracatal.functional_pt.pad import pad_2d

def ft_convolve(grid, kernel, default_dtype=torch.float32):

  if kernel.shape[1:] != grid.shape[1:]:
    padded_kernel = pad_2d(kernel, grid.shape)

  else:
    padded_kernel = kernel

  fourier_kernel = fft2(fftshift(padded_kernel, dim=(-2,-1)), dim=(-2,-1))
  fourier_grid = fft2(fftshift(grid, dim=(-2,-1)), dim=(-2,-1))
  fourier_product = fourier_grid * fourier_kernel
  real_spatial_convolved = torch.real(ifft2(fourier_product, dim=(-2,-1)))
  convolved = ifftshift(real_spatial_convolved, dim=(-2, -1))

  convolved = convolved.to(default_dtype)

  return convolved




