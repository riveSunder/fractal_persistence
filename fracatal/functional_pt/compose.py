import numpy as np
import numpy.random as npr
import torch

from fracatal.functional_pt.convolve import ft_convolve
from fracatal.functional_pt.pad import pad_2d

def make_gaussian(a, m, s, device=torch.device("cpu")):

  eps = torch.tensor([1e-9]).to(device)

  def gaussian(x):

    return a * torch.exp(-((x-m)/(eps+s))**2 / 2)


  return gaussian

def make_mixed_gaussian(amplitudes, means, std_devs):

  def gaussian_mixture(x):
    results = 0.0 * x
    eps = 1e-9 # prevent division by 0

    for a, m, s in zip(amplitudes, means, std_devs):
      my_gaussian = make_gaussian(a, m, s)
      results += my_gaussian(x)

    return results

  return gaussian_mixture

def make_kernel_field(kernel_radius, dim=122, default_dtype=torch.float32):

  x =  torch.arange(-dim / 2, dim / 2 + 1, 1)
  xx, yy = torch.meshgrid(x, x, indexing="ij")

  rr = torch.sqrt(xx**2 + yy**2) / kernel_radius
  rr = rr.to(default_dtype) 

  return rr
  
def make_make_kernel_function(amplitudes, means, standard_deviations, \
    dim=122, default_dtype=torch.float32, device=torch.device("cpu")):

  def make_kernel(kernel_radius):

    gm = make_mixed_gaussian(amplitudes, means, standard_deviations)
    rr = make_kernel_field(kernel_radius, dim=dim, default_dtype=default_dtype)

    kernel = gm(rr)[None,None,:,:]
    kernel = kernel / kernel.sum()

    return kernel.to(device)

  return make_kernel

def make_update_function(mean, standard_deviation, mode=0, device=torch.device("cpu")):
  # mode 0: use 2*f(x) -1
  # mode 1: use f(x)

  my_gaussian = make_gaussian(1.0, mean, standard_deviation, device=device)

  def lenia_update(x):
    """
    lenia growth function - mode 0
    """
    if mode == 0:
      return 2 * my_gaussian(x) - 1
    elif mode == 2:
      return my_gaussian(x) - 1
    else:
      return my_gaussian(x)

  return lenia_update

def make_update_step(update_function, kernel, dt, mode=0, \
    inner_kernel=None, persistence_function=None, \
    use_jit=False, clipping_function = lambda x: x, \
    default_dtype=torch.float32):

  def update_step(grid):

    neighborhoods = ft_convolve(grid, kernel)

    growth = update_function(neighborhoods)

    if persistence_function is None:
      dgrid_dt = growth
    elif inner_kernel is None:
      m = torch.clamp(grid, 0, 1)
      genesis = update_function(neighborhoods)
      persistence = persistence_function(neighborhoods)
      dgrid_dt = (1-m) * genesis + m * persistence
    else:
      m = ft_convolve(grid, inner_kernel, default_dtype=default_dtype)
      m = torch.clamp(m, 0, 1)
      genesis = update_function(neighborhoods)
      persistence = persistence_function(neighborhoods)

      dgrid_dt = (1-m) * genesis + m * persistence

    if mode == 1:
      # asymptotic update kawaguchi, suzuki, arita, chan 2020
      dgrid_dt = (dgrid_dt - grid)

    new_grid = clipping_function(grid + dt * dgrid_dt)

    return new_grid

  return update_step

