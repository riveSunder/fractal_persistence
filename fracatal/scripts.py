import numpy.random as npr
from jax import numpy as np
import skimage

from fracatal.functional_jax import make_update_step

import matplotlib
import matplotlib.pyplot as plt

def v_stability_sweep(pattern, make_kernel, my_update, \
            min_dt=0.001, max_dt=1.05, \
            min_kr=5, max_kr=51,k0=13,\
            parameter_steps=16,
            stride=4,\
            grid_dim=128,\
            persistence_update=None, \
            make_inner_kernel=None, \
            max_t=32, \
            max_steps=32000, \
            max_growth=2, \
            min_growth=0.5,\
            default_dtype = np.float16, \
            clipping_fn = lambda x: np.clip(x, 0.0, 1.0)):

  if stride > parameter_steps:
    print(f" stride {stride} greater than parameter ticks {parameter_steps}")
    print(f"\t\tUsing {parameter_steps} for stride instead")
    stride = min([stride, parameter_steps])
  
  dts = np.arange(min_dt, max_dt, (max_dt-min_dt) / parameter_steps, dtype=default_dtype)[:,None, None,None]
  krs = np.arange(min_kr, max_kr, (max_kr-min_kr) / parameter_steps, dtype=default_dtype)[:,None,None,None]
  
  results_img = np.zeros((dts.shape[0], krs.shape[0], 4), dtype=default_dtype)

  native_dim_h = pattern.shape[-2]
  native_dim_w = pattern.shape[-1]

  kernel = make_kernel(2)

  kr = krs#

  kernel = np.zeros((1, krs.shape[0],kernel.shape[-2], kernel.shape[-1]), dtype=default_dtype)
  patterns = np.zeros((dts.shape[0], krs.shape[0],kernel.shape[-2], kernel.shape[-1]), dtype=default_dtype)
  starting_grid = np.zeros((stride, krs.shape[0], grid_dim, grid_dim), dtype=default_dtype)
  
  for kk in range(kr.shape[0]):
    
    kernel = kernel.at[:,kk:kk+1,:,:].set(make_kernel(kr[kk].item()))
    scale_factor = kr[kk].item() / k0
    
    dim_h = int(native_dim_h * scale_factor)
    dim_w = int(native_dim_w * scale_factor)
    
    scaled_pattern = skimage.transform.resize(pattern, (1,1, dim_h, dim_w))
    
    starting_grid = starting_grid.at[:,kk:kk+1,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]].set(scaled_pattern)
   
  
  grid = starting_grid * 1.0

  full_dts = dts * 1.0

  explode = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  vanish = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  done = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  accumulated_t = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)
  total_steps = np.zeros((dts.shape[0], krs.shape[0],1,1), dtype=default_dtype)

  red_cmap = plt.get_cmap("Reds")
  green_cmap = plt.get_cmap("Greens")
  blue_cmap = plt.get_cmap("Blues")
  
  for jj in range(full_dts[::stride].shape[0]):
  
    dts = full_dts[jj*stride:(jj+1)*stride]
      
    grid = starting_grid * 1.0 
    
    starting_sum = grid.sum(axis=(2,3), keepdims=True)
    
    accumulated_t_part = np.zeros((grid.shape[0], grid.shape[1],1,1))
    total_steps_part = np.zeros((grid.shape[0], grid.shape[1],1,1))
    explode_part = np.zeros((grid.shape[0], grid.shape[1],1,1))
    vanish_part = np.zeros((grid.shape[0], grid.shape[1],1,1))
    
    results_img_part = np.zeros((grid.shape[0], grid.shape[1], 4))
    
    update_step = make_update_step(my_update, kernel, dts, clipping_fn)
    
    total_steps_counter = 0
    
    while accumulated_t_part.min() < max_t and total_steps_counter <= max_steps:
       
      grid = update_step(grid)
      
      g = grid.sum(axis=(2,3), keepdims=True) / starting_sum
    
      explode_part = explode_part + (g > max_growth)
      vanish_part = vanish_part + (g < min_growth)

      done_part = (explode_part + vanish_part) > 0
      
      accumulated_t_part += dts * (1 - done_part)
      total_steps_part += 1 * (1 - done_part)
      total_steps_counter += 1
     
    accumulated_t_truncated = np.clip(accumulated_t_part, 0, max_t)

    results_img_part = results_img_part.at[done_part.squeeze() <= 0].set(green_cmap(accumulated_t_truncated[done_part.squeeze() <= 0] / max_t).squeeze())  
    results_img_part = results_img_part.at[vanish_part.squeeze() > 0].set(blue_cmap(accumulated_t_truncated[vanish_part.squeeze() > 0] / max_t).squeeze())
    results_img_part = results_img_part.at[explode_part.squeeze() > 0].set(red_cmap(accumulated_t_truncated[explode_part.squeeze() > 0] / max_t).squeeze())
    
    results_img = results_img.at[jj*stride:(jj+1)*stride,:,:].set(results_img_part)

    accumulated_t = accumulated_t.at[jj*stride:(jj+1)*stride,:,:].set(accumulated_t_part)
    total_steps = total_steps.at[jj*stride:(jj+1)*stride,:,:].set(total_steps_part)

    explode = explode.at[jj*stride:(jj+1)*stride,:,:].set(explode_part)
    vanish = vanish.at[jj*stride:(jj+1)*stride,:,:].set(vanish_part)
    done = done.at[jj*stride:(jj+1)*stride,:,:].set(done_part)
    
  # accumulated_t can be larger than max_t when a batch contains different dt values
  results_img = np.array((255 * results_img), dtype=np.uint8)
  return results_img, accumulated_t, total_steps, explode, vanish, done


def stability_sweep(pattern, make_kernel, my_update, 
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
            clipping_fn = lambda x: np.clip(x, 0.0, 1.0)):

  if stride is not None:
    print(f" input argument stride (value: {stride}) not used in this algo")
  
  dts = np.arange(min_dt, max_dt, (max_dt-min_dt) / parameter_steps)[:,None, None,None]
  krs = np.arange(min_kr, max_kr, (max_kr-min_kr) / parameter_steps)[:,None,None,None]

  results_img = np.zeros((dts.shape[0], krs.shape[0], 4))

  native_dim_h = pattern.shape[-2]
  native_dim_w = pattern.shape[-1]

  explode = np.zeros((dts.shape[0], krs.shape[0],1,1))
  vanish = np.zeros((dts.shape[0], krs.shape[0],1,1))
  done = np.zeros((dts.shape[0], krs.shape[0],1,1))
  accumulated_t = np.zeros((dts.shape[0], krs.shape[0],1,1))
  total_steps = np.zeros((dts.shape[0], krs.shape[0],1,1))
  starting_grid = np.zeros((dts.shape[0], krs.shape[0], grid_dim, grid_dim))

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
            clipping_fn)
      else:
        update_step = make_update_step(my_update, kernel, dt, clipping_fn)

      accumulated_t_part = 0.0
      scale_factor = kr / k0

      dim_h = int(native_dim_h * scale_factor)
      dim_w = int(native_dim_w * scale_factor)

      scaled_pattern = skimage.transform.resize(pattern, (1,1, dim_h, dim_w))

      grid = starting_grid * 0.0
      grid = grid.at[:,:,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]].set(scaled_pattern)

      starting_sum = grid.sum()

      exploded = False
      vanished = False
      total_steps_counter = 0

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

  return results_img, accumulated_t, total_steps, explode, vanish, done

def v_stability_sweep_sl():
  
  # TODO: WIP!

  starting_grid = np.zeros((1,1,grid_dim, grid_dim))
  
  dts = np.arange(min_dt, max_dt, (max_dt-min_dt) / number_dt_steps)[:,None, None,None]
  krs = np.arange(min_kr, max_kr, (max_kr-min_kr) / number_kr_steps)[:,None,None,None]
  
  print(dts.shape, krs.shape)
  
  clipping_fn = lambda x: np.clip(x, 0.0, 1.0)
  results_img = np.zeros((dts.shape[0], krs.shape[0], 4))
  
  native_dim_h = pattern.shape[-2]
  native_dim_w = pattern.shape[-1]
  
  kernel_dim = 122
  make_kernel = make_make_kernel_function(amplitudes, means, standard_deviations, dim=kernel_dim)
  
  #make_kernel = make_make_smoothlife_kernel_function(r_inner, r_outer,dim = 126)
  #make_inner_kernel = make_make_smoothlife_kernel_function(0.0, r_inner, dim = 126)
  
  
  kernel = make_kernel(2)
  
  t0 = time.time()
  
  #my_update = gen
  #per_update = per
  
  for jj in range(krs[::stride].shape[0]):
    kr = krs[jj*stride:(jj+1)*stride]
      
    kernel = np.zeros((1, kr.shape[0],kernel.shape[-2], kernel.shape[-1]))
    inner_kernel = np.zeros((1, kr.shape[0],kernel.shape[-2], kernel.shape[-1]))
    patterns = np.zeros((dts.shape[0], kr.shape[0],kernel.shape[-2], kernel.shape[-1]))
    grid = np.zeros((dts.shape[0], kr.shape[0], starting_grid.shape[-2], starting_grid.shape[-1]))
            
    for kk in range(kr.shape[0]):
      
      kernel = kernel.at[:,kk:kk+1,:,:].set(make_kernel(kr[kk].item()))
      scale_factor = kr[kk].item() / k0
      
      dim_h = int(native_dim_h * scale_factor)
      dim_w = int(native_dim_w * scale_factor)
      
      scaled_pattern = skimage.transform.resize(pattern, (1,1, dim_h, dim_w))
      
      grid = grid.at[:,kk:kk+1,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]].set(scaled_pattern)
  
    
    if make_inner_kernel is not None:
      inner_kernel = inner_kernel.at[:,:,:,:].set(make_inner_kernel(k0))
      
    starting_sum = grid.sum(axis=(2,3), keepdims=True)
    
    explode = np.zeros((grid.shape[0], grid.shape[1],1,1))
    vanish = np.zeros((grid.shape[0], grid.shape[1],1,1))
    accumulated_t = np.zeros((grid.shape[0], grid.shape[1],1,1))
    total_steps = np.zeros((grid.shape[0], grid.shape[1],1,1))
    
    results_img_part = np.zeros((grid.shape[0], grid.shape[1], 4))
  
    if make_inner_kernel is not None:
      update_step = make_smoothlife_update_step(gen, per, \
          kernel, inner_kernel, dts, \
          clipping_function = lambda x: np.clip(x, 0, 1.0), \
          decimals=None)
  
    else:
      update_step = make_update_step(my_update, kernel, dts, clipping_fn)
  
    red_cmap = plt.get_cmap("Reds")
    green_cmap = plt.get_cmap("Greens")
    blue_cmap = plt.get_cmap("Blues")
    total_steps_counter = 0
    
    while accumulated_t.min() < max_t and total_steps_counter <= max_steps:
    
      grid = update_step(grid)
      
      g = grid.sum(axis=(2,3), keepdims=True) / starting_sum
    
      explode = explode + (g > max_growth)
      vanish = vanish + (g < min_growth)
      done = (explode + vanish) > 0
      
      accumulated_t += dts * (1 - done)
      total_steps += 1 * (1 - done)
      total_steps_counter += 1
    
    results_img_part = results_img_part.at[done.squeeze() <= 0].set(green_cmap(accumulated_t[done.squeeze() <= 0] / max_t).squeeze())
    results_img_part = results_img_part.at[explode.squeeze() > 0].set(red_cmap(accumulated_t[explode.squeeze() > 0] / max_t).squeeze())
    results_img_part = results_img_part.at[vanish.squeeze() > 0].set(blue_cmap(accumulated_t[vanish.squeeze() > 0] / max_t).squeeze())

    results_img = results_img.at[:,jj*stride:(jj+1)*stride,:].set(results_img_part)
  

  return results_img, accumulated_t, total_steps, explode, vanish, done
