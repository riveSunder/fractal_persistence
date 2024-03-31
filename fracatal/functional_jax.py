from jax import numpy as np
import numpy.random as npr

def pad_2d(kernel, pad_to):
  """
  pads the last two dimensions of grid to match dims of pad_to
  """

  if np.shape(kernel)[1:] != pad_to[1:]:

    diff_h  = pad_to[-2] - np.shape(kernel)[-2]
    diff_w =  pad_to[-1] - np.shape(kernel)[-1]
    pad_h = diff_h // 2
    pad_w = diff_w // 2

    rh, rw = diff_h % pad_h, diff_w % pad_w

    if rh:
      hp = rh
      hm = 0
    else:
      hp = 1
      hm = -1

    if rw:
      wp = rw
      wm = 0
    else:
      wp = 1
      wm = -1

    padded_kernel = np.pad(kernel, \
        ((0,0), (0,0), (pad_h+hp, pad_h+hm), (pad_w+wp, pad_w+wm)))
  else:
    return kernel

  return padded_kernel

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

def compute_diversity(subimage):
  """
  Computes Shannon diversity for pixel values in subimage
  """
  
  subimage = np.uint8(255*subimage / subimage.max())
  eps = 1e-9
  # compute Shannon diversity (aka entropy)
  p = np.zeros(256)
  
  for ii in range(p.shape[0]):
    p = p.at[ii].set(np.sum(subimage == ii))
      
  # normalize p
  p = p / p.sum()
  
  h = - np.sum( p * np.log2( eps+p))
  
  return h

def compute_frequency_ratio(subimage, ft_dim=65):
  rr = make_kernel_field(ft_dim, ft_dim-1)\

  ft_subimage = np.abs(np.fft.fftshift(np.fft.fft2(subimage, (ft_dim, ft_dim)))**2)

  frequency_ratio = (rr * ft_subimage).sum() / ((1.0 - rr) * ft_subimage).sum()

  return frequency_ratio

def compute_frequency_diversity(subimage, ft_dim=65):

  ft_subimage = np.abs(np.fft.fftshift(np.fft.fft2(subimage, (ft_dim, ft_dim)))**2)

  frequency_diversity = compute_diversity(ft_subimage)

  return frequency_diversity


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

  #dim = kernel_radius * 2 + 1

  x =  np.arange(-dim / 2, dim / 2 + 1, 1)
  xx, yy = np.meshgrid(x,x)

  rr = np.sqrt(xx**2 + yy**2) / kernel_radius

  rr = np.array(rr, dtype=default_dtype)
  return rr

def make_update_function(mean, standard_deviation):

  my_gaussian = make_gaussian(1.0, mean, standard_deviation)

  def lenia_update(x):
    """
    lenia update
    """
    return 2 * my_gaussian(x) - 1

  return lenia_update

def make_transition_function(mean, standard_deviation):

  my_gaussian = make_gaussian(1.0, mean, standard_deviation)

  def lenia_update(x):
    """
    lenia update
    """
    return my_gaussian(x) 

  return lenia_update

def make_update_step(update_function, kernel, dt, clipping_function = lambda x: x, default_dtype=np.float32):

  def update_step(grid):

    neighborhoods = ft_convolve(grid, kernel, default_dtype=default_dtype)
    dgrid_dt = update_function(neighborhoods)

    new_grid = clipping_function(grid + dt * dgrid_dt)

    return new_grid

  return update_step

def make_make_kernel_function(amplitudes, means, standard_deviations, \
    dim=126, default_dtype=np.float32):

  def make_kernel(kernel_radius):

    gm = make_mixed_gaussian(amplitudes, means, standard_deviations)
    rr = make_kernel_field(kernel_radius, dim=dim, default_dtype=default_dtype)

    kernel = gm(rr)[None,None,:,:]
    kernel = kernel / kernel.sum()

    return kernel

  return make_kernel

# smooth life
def sigmoid_1(x, mu, alpha, gamma=1):
  return 1 / (1 + np.exp(-4 * (x - mu) / alpha))

def get_smooth_steps_fn(intervals, alpha=0.0125):
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

