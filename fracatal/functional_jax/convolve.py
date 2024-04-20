from jax import numpy as np
import numpy.random as npr

from fracatal.functional_jax.pad import pad_2d

def ft_convolve(grid, kernel, default_dtype=np.float32):

  if np.shape(kernel)[1:] != np.shape(grid)[1:]:

    padded_kernel = pad_2d(kernel, np.shape(grid))

  else:                                     
    padded_kernel = kernel                          
                                        
  fourier_kernel = np.fft.fft2(np.fft.fftshift(padded_kernel, axes=(-2,-1)), axes=(-2,-1))
  fourier_grid = np.fft.fft2(np.fft.fftshift(grid, axes=(-2,-1)), axes=(-2,-1))
  fourier_product = fourier_grid * fourier_kernel 
  real_spatial_convolved = np.real(np.fft.ifft2(fourier_product, axes=(-2,-1)))
  convolved = np.fft.ifftshift(real_spatial_convolved, axes=(-2, -1))

  convolved = np.array(convolved, dtype=default_dtype)
                                        
  return convolved 
