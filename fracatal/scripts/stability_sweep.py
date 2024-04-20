import numpy.random as npr
from jax import numpy as np
import skimage

from fracatal.functional_jax import make_update_step, pad_2d

import matplotlib
import matplotlib.pyplot as plt

def stability_sweep(pattern, make_kernel, my_update, 
            dynamic_mode=0,
            min_dt=0.001, max_dt=1.05, \
            min_kr=5, max_kr=51, k0=13, \
            parameter_steps=16, \
            stride=None, \
            grid_dim=128, \
            persistence_update=None, \
            make_inner_kernel=None, \
            max_t=32, \
            max_steps=32000, \
            max_growth=2, \
            min_growth=0.5,\
            default_dtype=np.float16, \
            clipping_fn=lambda x: np.clip(x, 0.0, 1.0)):

  if stride is not None:
    print(f" input argument stride (value: {stride}) not used in this algo")
  
  dts = np.arange(min_dt, max_dt, (max_dt-min_dt) / parameter_steps, dtype=default_dtype)[:,None, None,None]
  krs = np.arange(min_kr, max_kr, (max_kr-min_kr) / parameter_steps, dtype=default_dtype)[:,None,None,None]

  results_img = np.zeros((dts.shape[0], krs.shape[0], 4))

  native_dim_h = pattern.shape[-2]
  native_dim_w = pattern.shape[-1]

  explode = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  vanish = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  done = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  accumulated_t = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  total_steps = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  #starting_grid = np.zeros((dts.shape[0], krs.shape[0], grid_dim, grid_dim), dtype=default_dtype)
  starting_grid = np.zeros((1, 1, grid_dim, grid_dim), dtype=default_dtype)

  red_cmap = plt.get_cmap("Reds")
  green_cmap = plt.get_cmap("Greens")
  blue_cmap = plt.get_cmap("Blues")

  for ii, dt in enumerate(dts):
    for jj, kr in enumerate(krs):

      kernel = make_kernel(kr.item())

      if make_inner_kernel is not None:
        inner_kernel = make_inner_kernel(k0)
        update_step = make_smoothlife_update_step(my_update, persistence_update, \
            kernel, inner_kernel, dt, \
            clipping_fn, \
            mode=dynamic_mode, \
            )

      else:
        update_step = make_update_step(my_update, kernel, dt, clipping_function=clipping_fn, \
            mode=dynamic_mode, \
            default_dtype=default_dtype)

      scale_factor = kr.item() / k0

      if scale_factor < 1.0:
        scaled_pattern = np.array(skimage.transform.rescale(pattern, (1,1, scale_factor, scale_factor), order=5, anti_aliasing=True), \
          dtype=default_dtype)
      else:
        scaled_pattern = np.array(skimage.transform.rescale(pattern, (1,1, scale_factor, scale_factor), order=5), dtype=default_dtype)

      starting_grid = starting_grid.at[:,:,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]].set(scaled_pattern)

      grid = starting_grid * 0.0
      grid = grid.at[:,:,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]].set(scaled_pattern)

      if grid.shape[1:] != kernel.shape[1:]:
        if jj == 0 and ii == 0: 
            print(f"pre-padding kernel")
        kernel = pad_2d(kernel, grid.shape)

      starting_sum = grid.sum()

      exploded = False
      vanished = False
      total_steps_counter = 0
      accumulated_t_part = 0.0

      grid_0 = 1.0 * grid
      while accumulated_t_part < max_t and total_steps_counter <= max_steps:

        grid = update_step(grid)

        g = grid.sum() / starting_sum

        accumulated_t_part += dt
        total_steps_counter += 1

        if g > max_growth:
          exploded = True
          break
        if g < min_growth:
          vanished = True
          break

      accumulated_t_part = accumulated_t_part.item()
      accumulated_t_truncated = np.clip(accumulated_t_part, 0, max_t)

      if exploded == True:
        results_img = results_img.at[ii,jj].set(red_cmap(accumulated_t_truncated / max_t))
      elif vanished == True:
        results_img = results_img.at[ii,jj].set(blue_cmap(accumulated_t_truncated / max_t))
      else:
        results_img = results_img.at[ii,jj].set(green_cmap(accumulated_t_truncated / max_t))

      accumulated_t = accumulated_t.at[ii,jj].set(accumulated_t_part)
      total_steps = total_steps.at[ii,jj].set(total_steps_counter)
      explode = explode.at[ii,jj].set(exploded)
      vanish = vanish.at[ii,jj].set(vanished)
      done = done.at[ii,jj].set(exploded or vanished)

  results_img = np.array((255 * results_img), dtype=np.uint8)

  return results_img, accumulated_t, total_steps, explode, vanish, done, grid_0, grid

