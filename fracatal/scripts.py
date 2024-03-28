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
                      max_t = 64, \
                      max_steps = 64000, \
                      max_growth=2, \
                      min_growth=0.5,\
                      clipping_fn = lambda x: np.clip(x, 0.0, 1.0)):
    if stride > parameter_steps:
      print(f" stride {stride} greater than parameter ticks {parameter_steps}")
      print(f"\t\tUsing {parameter_steps} for stride instead")
      stride = min([stride, parameter_steps])
    
    dts = np.arange(min_dt, max_dt, (max_dt-min_dt) / parameter_steps)[:,None, None,None]
    krs = np.arange(min_kr, max_kr, (max_kr-min_kr) / parameter_steps)[:,None,None,None]
    
    results_img = np.zeros((dts.shape[0], krs.shape[0], 4))

    native_dim_h = pattern.shape[-2]
    native_dim_w = pattern.shape[-1]

    kernel = make_kernel(2)

    kr = krs#

    kernel = np.zeros((1, krs.shape[0],kernel.shape[-2], kernel.shape[-1]))
    patterns = np.zeros((dts.shape[0], krs.shape[0],kernel.shape[-2], kernel.shape[-1]))
    starting_grid = np.zeros((stride, krs.shape[0], grid_dim, grid_dim))
    
    for kk in range(kr.shape[0]):
        
        kernel = kernel.at[:,kk:kk+1,:,:].set(make_kernel(kr[kk].item()))
        scale_factor = kr[kk].item() / k0
        
        dim_h = int(native_dim_h * scale_factor)
        dim_w = int(native_dim_w * scale_factor)
        
        scaled_pattern = skimage.transform.resize(pattern, (1,1, dim_h, dim_w))
        
        starting_grid = starting_grid.at[:,kk:kk+1,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]].set(scaled_pattern)
   
    
    grid = starting_grid * 1.0

    dtss = dts * 1.0
    
    for jj in range(dtss[::stride].shape[0]):
    
        dts = dtss[jj*stride:(jj+1)*stride]
            
        grid = starting_grid * 1.0 #np.zeros((dts.shape[0], kr.shape[0], starting_grid.shape[-2], starting_grid.shape[-1]))
        
        starting_sum = grid.sum(axis=(2,3), keepdims=True)
        
        explode = np.zeros((grid.shape[0], grid.shape[1],1,1))
        vanish = np.zeros((grid.shape[0], grid.shape[1],1,1))
        accumulated_t = np.zeros((grid.shape[0], grid.shape[1],1,1))
        total_steps = np.zeros((grid.shape[0], grid.shape[1],1,1))
        
        results_img_part = np.zeros((grid.shape[0], grid.shape[1], 4))
        
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
            #finished = accumulated_t > max_t
            done = (explode + vanish) > 0
            
            accumulated_t += dts * (1 - done)
            total_steps += 1 * (1 - done)
            total_steps_counter += 1
         
        results_img_part = results_img_part.at[done.squeeze() <= 0].set(green_cmap(accumulated_t[done.squeeze() <= 0] / max_t).squeeze())  
        results_img_part = results_img_part.at[explode.squeeze() > 0].set(red_cmap(accumulated_t[explode.squeeze() > 0] / max_t).squeeze())
        results_img_part = results_img_part.at[vanish.squeeze() > 0].set(blue_cmap(accumulated_t[vanish.squeeze() > 0] / max_t).squeeze())
        
        results_img = results_img.at[jj*stride:(jj+1)*stride,:,:].set(results_img_part)
        
    return results_img, accumulated_t, total_steps, explode, vanish, done

def stability_sweep(dts, krs, starting_grid, pattern, make_kernel, my_update, max_t, \
                    persistence_update=None, \
                    make_inner_kernel=None, \
                    max_steps=1000, \
                    max_growth=2, \
                    min_growth=0.5, \
                    k0=31):


    clipping_fn = lambda x: np.clip(x, 0.0, 1.0)
    results_img = np.zeros((dts.shape[0], krs.shape[0],3))
    max_growth = 1.5
    min_growth = 0.5


    native_dim_h = pattern.shape[-2]
    native_dim_w = pattern.shape[-1]

    for ii, dt in enumerate(dts):
        for jj, kr in enumerate(krs):

            kernel = make_kernel(kr)

            if make_inner_kernel is not None:
                inner_kernel = make_inner_kernel(k0)
                update_step = make_smoothlife_update_step(my_update, persistence_update, \
                        kernel, inner_kernel, dt, \
                        clipping_fn)
            else:
                update_step = make_update_step(my_update, kernel, dt, clipping_fn)

            accumulated_t = 0.0
            total_steps = 0
            scale_factor = kr / k0

            dim_h = int(native_dim_h * scale_factor)
            dim_w = int(native_dim_w * scale_factor)

            scaled_pattern = skimage.transform.resize(pattern, (1,1, dim_h, dim_w))

            grid = starting_grid * 0.0
            grid[:,:,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]] = scaled_pattern

            starting_sum = grid.sum()

            explode = False
            vanish = False

            if(0):

                plt.figure()
                plt.imshow(scaled_pattern.squeeze())
                plt.show()

                plt.figure()
                plt.imshow(grid.squeeze())
                plt.show()

            while accumulated_t < max_t and total_steps <= max_steps:

                grid = update_step(grid)

                g = grid.sum() / starting_sum

                accumulated_t += dt
                total_steps += 1
                if g > max_growth:
                    explode = True
                    break
                if g < min_growth:
                    vanish = True
                    break
            if(0):
                plt.figure()
                plt.imshow(grid.squeeze())
                plt.show()

            if explode == True:
                results_img[ii,jj,0] = 1-accumulated_t / max_t
            elif vanish == True:
                results_img[ii,jj,2] = 1-accumulated_t / max_t
            else:
                results_img[ii,jj,1] = accumulated_t / max_t

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
        green_cmap = plt.get_cmap("Purples")
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
