import numpy as np
import torch

from torch.fft import fftshift, fft2

from fracatal.functional_pt.compose import make_kernel_field

def compute_entropy(subimage):
  """
  Computes Shannon entropy for pixel values in subimage
  """
  
  subimage = torch.tensor(255*subimage.clone() / subimage.max(), dtype=torch.uint8)

  eps = 1e-9
  # compute Shannon entropy 
  p = torch.zeros(256)
  
  for ii in range(p.shape[0]):
    p[ii] = torch.sum(subimage == ii)
      
  # normalize p
  p = p / p.sum()
  
  h = - torch.sum( p * torch.log2( eps+p))
  
  return h

def compute_frequency_ratio(subimage, ft_dim=65):
  eps= 1e-9
  rr = make_kernel_field(ft_dim, ft_dim-1)\

  ft_subimage = torch.abs(fftshift(fft2(subimage, (ft_dim, ft_dim)))**2)

  frequency_ratio = (rr * ft_subimage).sum() / (eps + (1.0 - rr) * ft_subimage).sum()

  return frequency_ratio

def compute_frequency_entropy(subimage, ft_dim=65):

  ft_subimage = torch.abs(fftshift(fft2(subimage, (ft_dim, ft_dim)))**2)

  frequency_entropy = compute_entropy(ft_subimage)

  return frequency_entropy
