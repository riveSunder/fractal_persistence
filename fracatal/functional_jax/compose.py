import jax
from jax import numpy as np
import numpy.random as npr

from fracatal.functional_jax.convolve import ft_convolve
from fracatal.functional_jax.pad import pad_2d

def make_gaussian(a, m, s):

  eps = 1e-9

  def gaussian(x):

    return a * np.exp(-((x-m)/(eps+s))**2 / 2)

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

def make_kernel_field(kernel_radius, dim=126, default_dtype=np.float32):

  x =  np.arange(-dim / 2, dim / 2 + 1, 1)
  xx, yy = np.meshgrid(x,x)

  rr = np.sqrt(xx**2 + yy**2) / kernel_radius

  rr = np.array(rr, dtype=default_dtype)
  return rr

def make_make_kernel_function(amplitudes, means, standard_deviations, \
    dim=122, default_dtype=np.float32):

  def make_kernel(kernel_radius):

    gm = make_mixed_gaussian(amplitudes, means, standard_deviations)
    rr = make_kernel_field(kernel_radius, dim=dim, default_dtype=default_dtype)

    kernel = gm(rr)[None,None,:,:]
    kernel = kernel / kernel.sum()

    return kernel

  return make_kernel
  
def make_update_function(mean, standard_deviation, mode=0, use_jit=False):
  # mode 0: use 2*f(x) -1
  # mode 1: use f(x)

  my_gaussian = make_gaussian(1.0, mean, standard_deviation)

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

  if use_jit:
    return jax.jit(lenia_update)
  else:
    return lenia_update


def make_update_step(update_function, kernel, dt, mode=0, \
    inner_kernel=None, persistence_function=None, \
    use_jit=False, clipping_function = lambda x: x, \
    default_dtype=np.float32):


  def jit_convolve(grid, kernel):
    return ft_convolve(grid, kernel, default_dtype=default_dtype)

  if use_jit:
    my_convolve = jax.jit(jit_convolve)
  else:
    my_convolve = jit_convolve

  def update_step(grid):

    neighborhoods = my_convolve(grid, kernel)

    growth = update_function(neighborhoods)

    if persistence_function is None:
      dgrid_dt = growth
    elif inner_kernel is None:
      m = np.clip(grid, 0, 1)
      genesis = update_function(neighborhoods)
      persistence = persistence_function(neighborhoods)
      dgrid_dt = (1-m) * genesis + m * persistence
    else:
      m = ft_convolve(grid, inner_kernel, default_dtype=default_dtype)
      m = np.clip(m, 0, 1)
      genesis = update_function(neighborhoods)
      persistence = persistence_function(neighborhoods)

      dgrid_dt = (1-m) * genesis + m * persistence

    if mode == 1:
      # asymptotic update kawaguchi, suzuki, arita, chan 2020
      dgrid_dt = (dgrid_dt - grid)

    new_grid = clipping_function(grid + dt * dgrid_dt)

    return new_grid

  if use_jit:
    return jax.jit(update_step)
  else:
    return update_step


# smooth life
def sigmoid_1(x, mu, alpha, gamma=1):
  return 1 / (1 + np.exp(-4 * (x - mu) / alpha))

def make_smooth_steps_function(intervals, alpha=0.0125):
  """
  construct an update function from intervals.
  input intervals is a list of lists of interval bounds,
  each element contains the start and end of an interval.

  # this code was adopated from rivesunder/yuca
  """

  def smooth_steps_fn(x):

    result = np.zeros_like(x)
    for bounds in intervals:
      result += sigmoid_1(x, bounds[0], alpha) * (1 - sigmoid_1(x, bounds[1], alpha))

    return result

  return smooth_steps_fn

def make_make_smoothlife_kernel_function(r_inner, r_outer, dim=126):
  """
  r_inner and r_outer are 1/3. and 1.05 for Rafler's SmoothLife glider configuration
  """

  def make_smoothlife_kernel(kernel_radius=10):

    rr = make_kernel_field(kernel_radius, dim=dim)
    kernel = np.ones_like(rr)
    kernel = kernel.at[rr < r_inner].set(0.0)
    kernel = kernel.at[rr >= r_outer].set(0.0)

    kernel = (kernel / kernel.sum())[None,None,:,:]

    return kernel

  return make_smoothlife_kernel

def make_smooth_interval(alpha=0.1470, intervals=[[0.2780, 0.3650]]):

  def smooth_interval(x):

    result = 0.0 * x

    for bounds in intervals:
      result += sigmoid_1(x, bounds[0], alpha) * (1-sigmoid_1(x, bounds[1], alpha))

    return 2 * result - 1

  return smooth_interval

def make_smoothlife_update_function(intervals=[[0.2780, 0.3650]], \
                  alpha=0.028):

  smooth = make_smooth_interval(alpha=alpha, intervals=intervals)

  def smoothlife_update(x):
    """
    smoothlife half-update
    """
    return 2 * smooth(x) - 1

  return smoothlife_update

def make_smoothlife_update_step(genesis_function, persistence_function, \
                kernel, inner_kernel, dt, clipping_function = lambda x: x, \
                decimals=None):

  if decimals is not None:
    r = lambda x: np.round(x, decimals=decimals)
  else:
    r = lambda x: x

  def update_step(grid):

    neighborhoods = r(ft_convolve(r(grid), r(kernel)))
    inner = r(ft_convolve(r(grid), r(inner_kernel)))

    genesis = r(genesis_function(neighborhoods))
    persistence = r(persistence_function(neighborhoods))
    dgrid_dt = (1-inner) * genesis + inner * persistence

    new_grid = r(clipping_function(r(grid) + dt * dgrid_dt))

    return new_grid

  return update_step

