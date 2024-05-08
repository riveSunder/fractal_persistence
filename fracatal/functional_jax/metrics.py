from jax import numpy as np
import numpy.random as npr

from fracatal.functional_jax.compose import make_gaussian, \
        make_mixed_gaussian, \
        make_kernel_field, \
        make_update_function, \
        make_update_step, \
        make_make_kernel_function, \
        sigmoid_1, \
        make_smooth_steps_function, \
        make_make_smoothlife_kernel_function, \
        make_smooth_interval, \
        make_smoothlife_update_function, \
        make_smoothlife_update_step

def compute_entropy(subimage):
  """
  Computes Shannon entropy for pixel values in subimage
  """
  
  subimage = np.uint8(255*subimage / subimage.max())
  eps = 1e-9
  # compute Shannon entropy 
  p = np.zeros(256)
  
  for ii in range(p.shape[0]):
    p = p.at[ii].set(np.sum(subimage == ii))
      
  # normalize p
  p = p / p.sum()
  
  h = - np.sum( p * np.log2( eps+p))
  
  return h

def compute_frequency_ratio(subimage, ft_dim=65):
  eps= 1e-9
  rr = make_kernel_field(ft_dim, ft_dim-1)\

  ft_subimage = np.abs(np.fft.fftshift(np.fft.fft2(subimage, (ft_dim, ft_dim)))**2)

  frequency_ratio = (rr * ft_subimage).sum() / (eps + (1.0 - rr) * ft_subimage).sum()

  return frequency_ratio

def compute_frequency_entropy(subimage, ft_dim=65):

  ft_subimage = np.abs(np.fft.fftshift(np.fft.fft2(subimage, (ft_dim, ft_dim)))**2)

  frequency_entropy = compute_entropy(ft_subimage)

  return frequency_entropy
