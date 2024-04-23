import sys
import argparse
import subprocess
import os

import numpy.random as npr
from jax import numpy as np
import skimage


import time

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
matplotlib.rcParams["animation.embed_limit"] = 1024

import skimage
import skimage.io as sio
import skimage.transform
import fracatal

from fracatal.functional_jax.convolve import ft_convolve
from fracatal.functional_jax.pad import pad_2d
from fracatal.functional_jax.metrics import compute_entropy, \
        compute_frequency_ratio, \
        compute_frequency_entropy
from fracatal.functional_jax.compose import make_gaussian, \
        make_mixed_gaussian, \
        make_kernel_field, \
        make_update_function, \
        make_update_step, \
        make_make_kernel_function, \
        sigmoid_1, \
        get_smooth_steps_fn, \
        make_make_smoothlife_kernel_function, \
        make_smooth_interval, \
        make_smoothlife_update_function, \
        make_smoothlife_update_step

import IPython

from mpi4py import MPI
comm = MPI.COMM_WORLD

def mpi_fork(workers):
  """
  relaunches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  via https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease
  """
  global worker_number, rank

  if workers <= 1:
      print("no workers, n<=1")
      worker_number = 0
      rank = 0
      return "child"

  if os.getenv("IN_MPI") is None:
      env = os.environ.copy()
      env.update(\
              MKL_NUM_THREADS="1", \
              OMP_NUM_THREAdS="1",\
              IN_MPI="1",\
              )
      print( ["mpirun", "-np", str(workers), sys.executable]  + sys.argv)
      subprocess.check_call(["mpirun", "-np", str(workers), sys.executable] + ['-u']+ sys.argv, env=env)

      return "parent"
  else:
      worker_number = comm.Get_size()
      rank = comm.Get_rank()
      return "child"

def mpi_stability_sweep(pattern, make_kernel, \
            dynamic_mode=0,\
            min_mu=0.15, max_mu=None, \
            min_sigma=0.017, max_sigma=None, \
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
            clipping_fn=lambda x: np.clip(x, 0.0, 1.0), \
            verbosity=0,\
            workers=8):

  if mpi_fork(workers) == "parent":
      os._exit(0)

  if rank == 0:
      results = mantle(pattern, make_kernel,  \
            dynamic_mode,
            min_mu, max_mu, \
            min_sigma, max_sigma, \
            min_dt, max_dt, \
            min_kr, max_kr, k0, \
            parameter_steps, \
            stride, \
            grid_dim, \
            persistence_update, \
            make_inner_kernel, \
            max_t, \
            max_steps, \
            max_growth, \
            min_growth,\
            default_dtype, \
            verbosity,\
            clipping_fn, workers)
      return results
  else:
      arm(pattern, make_kernel,  \
            dynamic_mode,
            min_mu, max_mu, \
            min_sigma, max_sigma, \
            min_dt, max_dt, \
            min_kr, max_kr, k0, \
            parameter_steps, \
            stride, \
            grid_dim, \
            persistence_update, \
            make_inner_kernel, \
            max_t, \
            max_steps, \
            max_growth, \
            min_growth,\
            default_dtype, \
            verbosity,\
            clipping_fn)


def mantle(pattern, make_kernel, \
            dynamic_mode=0,\
            min_mu=0.15, max_mu=None, \
            min_sigma=0.017, max_sigma=None, \
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
            verbosity=0, \
            clipping_fn=lambda x: np.clip(x, 0.0, 1.0),\
            workers=1):

  if stride is not None:
    print(f" input argument stride (value: {stride}) not used in this algo")
  # call from root_dir
  # future update use __file__ to make relative path to root_dir
  root_dir = "."

  t0 = time.time()

  max_zooms = 20
  total_zooms = 0

  time_elapsed = time.time()-t0
  time_stamp = str(t0).replace(".","_") #int(t0*1000)

  freq_zoom_strides = 5
  freq_zoom_fraction = 2
  idx = 0

  exp_name = f"{pattern_name}_{tag}_{time_stamp}"
  save_dir = os.path.join(root_dir, "results", exp_name)

  if os.path.exists(save_dir):
      pass
  else:
      os.mkdir(save_dir)

  metadata_path = None
  params = [min_kr, max_kr, min_dt, max_dt]

  while time_elapsed <= max_runtime and total_zooms <= max_zooms:
      
    dynamic_mode = 1 if "asym" in pattern_name else 0
    t1 = time.time()

    # hierarchy of parameters mu > sigma > dt > kr
    # in other words, if max_mu and max_sigma are both specified, dt and kr are static

    if max_mu is not None:
      mus = np.arange(min_mu, max_mu, (max_mu-min_mu) / parameter_steps, dtype=default_dtype)[:,None, None,None]
      active_mu = True
    else:
      mus = np.array([min_mu])[:,None,None,None]
      active_mu = False

    if max_sigma is not None:
      sigmas = np.arange(min_sigma, max_sigma, (max_sigma-min_sigma) / parameter_steps, dtype=default_dtype)[:,None, None,None]
      active_sigma = True
    else:
      sigmas = np.array([min_sigma])[:,None,None,None]
      active_sigma = False

    if (not(active_mu and active_sigma)) and max_dt is not None:
      dts = np.arange(min_dt, max_dt, (max_dt-min_dt) / parameter_steps, dtype=default_dtype)[:,None, None,None]
      active_dt = True
    else:
      dts = np.array([min_dt])[:,None,None,None]
      active_dt = False
      
    if (not(active_mu or active_sigma) or not(active_dt)) and max_kr is not None:
      krs = np.arange(min_kr, max_kr, (max_kr-min_kr) / parameter_steps, dtype=default_dtype)[:,None,None,None]
      active_kr = True
    else:
      krs = np.array([min_kr])[:,None,None,None]
      active_kr = False

    assert np.sum(np.array([active_kr, active_dt, active_sigma, active_mu])) == 2, "only two parameters should have ranges"
      
    results_img = np.zeros((parameter_steps, parameter_steps, 4))

    native_dim_h = pattern.shape[-2]
    native_dim_w = pattern.shape[-1]

    explode = np.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    vanish = np.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    done = np.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    accumulated_t = np.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    total_steps = np.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    #starting_grid = np.zeros((dts.shape[0], krs.shape[0], grid_dim, grid_dim), dtype=default_dtype)
    starting_grid = np.zeros((1, 1, grid_dim, grid_dim), dtype=default_dtype)

    red_cmap = plt.get_cmap("Reds")
    green_cmap = plt.get_cmap("Greens")
    blue_cmap = plt.get_cmap("Blues")

    run_index = 0
    run_max = parameter_steps**2 #dts.shape[0] * krs.shape[0]
    #print(f"run max {run_max}")

    while run_index < run_max:
        ## parallel mpi section ****
        last_worker = 1 * workers
        for worker_idx in range(1, workers):
          if run_index < run_max:
            comm.send((run_index, mus, sigmas, dts, krs, [active_mu, active_sigma, active_dt, active_kr]), dest=worker_idx)
            #print(f"run index {run_index} sent to worker {worker_idx}")
            run_index += 1
          else:
            last_worker = worker_idx
            break

          

        for worker_idx in range(1, last_worker):

          if verbosity: print(f"rec'ing from worker {worker_idx} of {last_worker-1}")

          results_part = comm.recv(source=worker_idx)
          
          run_index_part = results_part[0]
          accumulated_t_part = results_part[1]
          total_steps_part = results_part[2]
          exploded = results_part[3]
          vanished = results_part[4]
          grid_0 = results_part[5]
          grid = results_part[6]

          mu = results_part[7]
          sigma = results_part[8]
          dt = results_part[9]
          kr = results_part[10]

          ii = int(np.floor(run_index_part / parameter_steps))
          jj = run_index_part % parameter_steps

          param_indices = [0,0,0,0]
          active_list = [active_mu, active_sigma, active_dt, active_kr]
          count_active = 0

          for ll in range(len(param_indices)):
            if active_list[ll] and count_active == 0:
              param_indices[ll] = ii
              count_active += 1
            elif active_list[ll] and count_active == 1:
              param_indices[ll] = jj
              count_active += 1

            if active_list[ll]:
              if count_active == 1:
                if ll == 0:
                  y_ticks = 1.0 * mus
                  ylabel = "$\mu$"
                  min_y = min_mu
                  max_y = max_mu
                elif ll == 1:
                  y_ticks = 1.0 * sigmas
                  ylabel = "$\sigma$"
                  min_y = min_sigma
                  max_y = max_sigma
                elif ll == 2:
                  y_ticks = 1.0 * dts
                  ylabel = "$\Delta t$"
                  min_y = min_dt
                  max_y = max_dt
              elif count_active == 2:
                if ll == 1:
                  x_ticks = 1.0 * sigmas
                  xlabel = "$\sigma$"
                  min_x = min_sigma
                  max_x = max_sigma
                elif ll == 2:
                  x_ticks = 1.0 * dts
                  xlabel = "$\Delta t$"
                  min_x = min_dt
                  max_x = max_dt
                elif ll == 3:
                  x_ticks = 1.0 * krs
                  xlabel = "$k_r$"
                  min_x = min_kr
                  max_x = max_kr
                break

          mu_check = mus[param_indices[0]]
          sigma_check = sigmas[param_indices[1]]
          dt_check = dts[param_indices[2]]
          kr_check = krs[param_indices[3]]

          if not(np.isclose(mu_check, mu)): 
            print("dt rec'd from worker does not match")
            print(f"{mu_check}, {mu}, {worker_idx}")
          if not(np.isclose(sigma_check, sigma)):
            print("dt rec'd from worker does not match")
            print(f"{sigma_check}, {sigma}, {worker_idx}")
          if not(np.isclose(dt_check, dt)):
            print("dt rec'd from worker does not match")
            print(f"{dt_check}, {dt}, {worker_idx}")
          if not(np.isclose(kr_check, kr)):
            print("kr rec'd from worker does not match")
            print(f"{kr_check}, {kr}, {worker_idx}")

          assert mu == mu_check, f"expected mu={mu_check}, got {mu}"
          assert sigma == sigma_check,  f"expected sigma={sigma_check}, got {sigma}"
          assert dt == dt_check,  f"expected mu={dt_check}, got {dt}"
          assert kr == kr_check,  f"expected mu={kr_check}, got {kr}"

          accumulated_t_part = accumulated_t_part.item()
          accumulated_t_truncated = np.clip(accumulated_t_part, 0, max_t)

          if exploded == True:
            results_img = results_img.at[ii,jj].set(red_cmap(accumulated_t_truncated / max_t))
          elif vanished == True:
            results_img = results_img.at[ii,jj].set(blue_cmap(accumulated_t_truncated / max_t))
          else:
            results_img = results_img.at[ii,jj].set(green_cmap(accumulated_t_truncated / max_t))

          accumulated_t = accumulated_t.at[ii,jj].set(accumulated_t_part)
          total_steps = total_steps.at[ii,jj].set(total_steps_part)
          explode = explode.at[ii,jj].set(exploded)
          vanish = vanish.at[ii,jj].set(vanished)
          done = done.at[ii,jj].set(exploded or vanished)


    results_img = np.array((255 * results_img), dtype=np.uint8)

    results = [[results_img, accumulated_t, total_steps, explode, vanish, done, grid_0, grid]]

    t2 = time.time()

    fig, ax = plt.subplots(1,1, figsize=(12,12))
    ax.imshow(results[-1][0])
    
    number_ticklabels = min([16, parameter_steps])
    ticklabel_period = parameter_steps // number_ticklabels
    yticklabels = [f"{elem.item():.6e}" if not(mm % ticklabel_period) else "" for mm, elem in enumerate(y_ticks)]
    xticklabels = [f"{elem.item():.6e}" if not(mm % ticklabel_period) else "" for mm, elem in enumerate(x_ticks)]
    
    _ = ax.set_yticks(np.arange(0,y_ticks.shape[0]))
    _ = ax.set_yticklabels(yticklabels, fontsize=16,  rotation=0)
    _ = ax.set_xticks(np.arange(0,x_ticks.shape[0]))
    _ = ax.set_xticklabels(xticklabels, fontsize=16, rotation=90)
    _ = ax.set_ylabel(ylabel, fontsize=22)
    _ = ax.set_xlabel(xlabel, fontsize=22)
    

    msg2 = f"total elapsed: {t2-t0:.3f} s, last sweep: {t2-t1:.3f}\n"
    msg = f"    {xlabel} from {min_x:.2e} to {max_x:.2e}\n"
    msg += f"   {ylabel} from {min_y:2e} to {max_y:.2e}\n"
    
    ax.set_title("disco persistence \n" + msg, fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{root_dir}/assets/disco{time_stamp}_{idx}.png")
    #plt.show() 
       
    print(msg2 + msg)
    # save results
    # results_img, accumulated_t, total_steps, explode, vanish, done, grid_0, grid
    
    img_savepath = os.path.join(save_dir, f"{exp_name}_img_{idx}.png")
    img_npy_savepath = os.path.join(save_dir, f"{exp_name}_img_{idx}.npy")
    
    accumulated_t_savepath = os.path.join(save_dir, f"{exp_name}_accumulated_t_{idx}.npy")
    total_steps_savepath = os.path.join(save_dir, f"{exp_name}_total_steps_{idx}.npy")
    
    explode_savepath = os.path.join(save_dir, f"{exp_name}_explode_{idx}.npy")
    vanish_savepath = os.path.join(save_dir, f"{exp_name}_vanish_{idx}.npy")
    done_savepath = os.path.join(save_dir, f"{exp_name}_done_{idx}.npy")
    
    grid_0_savepath = os.path.join(save_dir, f"{exp_name}_grid_0_{idx}.npy")
    grid_T_savepath = os.path.join(save_dir, f"{exp_name}_grid_T_{idx}.npy")
                                    
    sio.imsave(img_savepath, results[-1][0])
    np.save(img_npy_savepath, results[-1][0])
    np.save(accumulated_t_savepath, results[-1][1])
    np.save(total_steps_savepath, results[-1][2])
    np.save(explode_savepath, results[-1][3])
    np.save(vanish_savepath, results[-1][4])
    np.save(done_savepath, results[-1][5])
    np.save(grid_0_savepath, results[-1][6])
    np.save(grid_T_savepath, results[-1][7])
    
    # log experiment metadata
    #metadata = "index, min_dt, max_dt, min_kr, max_kr, parameter_steps, time_stamp, "
    #metadata += "img_savepath, accumulated_t_savepath, total_steps_savepath, explode_savepath, vanish_savepath, grid_T_savepath\n"

    if metadata_path is None:
      metadata_path = os.path.join(save_dir, f"metadata_{time_stamp}.txt")
      metadata = "index, pattern_name, min_dt, max_dt, min_kr, max_kr, parameter_steps, max_t, max_steps, max_runtime, "
      metadata += "time_stamp, sim_time_elapsed,  total_time_elapsed, "
      metadata += "img_savepath, accumulated_t_savepath, total_steps_savepath, explode_savepath, vanish_savepath, grid_T_savepath\n"

      with open(metadata_path,"w") as f:
          f.write(metadata)

    metadata = f"{idx}, {pattern_name}, {min_dt}, {max_dt}, {min_kr}, {max_kr}, {parameter_steps}, {max_t}, {max_steps}, {max_runtime}, "
    metadata += f"{time_stamp}, {t2-t1:2f}, {t2-t0:2f}, "
    metadata += f"{img_savepath}, {accumulated_t_savepath}, {total_steps_savepath}, {explode_savepath}, {vanish_savepath}, {grid_T_savepath}\n"
    with open(metadata_path,"a") as f:
        f.write(metadata)
    # determine next parameter range
    if time.time()-t0 < max_t:
      freq_zoom_dim = (results[-1][5].shape[1]) // freq_zoom_fraction
      freq_zoom_stride = 4 + int(parameter_steps/4)
      freq_zoom_strides = (results[-1][5].shape[1]-freq_zoom_dim) // freq_zoom_stride +1
      
      fzd = freq_zoom_dim
      fzs = freq_zoom_stride
      
      params_list = []
      entropy = []
      frequency_entropy = []
      frequency_ratio = []
      # Weighted RGB conversion to grayscale
      gray_image = (1.0 - results[-1][5])
              #0.29 * results[-1][0][:,:,0] \
              #+ 0.6*results[-1][0][:,:,1] \
              #+ 0.11 * results[-1][0][:,:,2]  
      
      for ll in range(freq_zoom_strides**2):
          fzd = freq_zoom_dim
          fzs = freq_zoom_stride
          
          cx = int(np.floor(ll / freq_zoom_strides))
          cy = ll % freq_zoom_strides
          
          params_list.append([x_ticks[cy*fzs].item(), \
                  x_ticks[cy*fzs+fzd].item(),\
                  y_ticks[cx*fzs].item(), \
                  y_ticks[cx*fzs+fzd].item()])


          subimage = gray_image[cx*fzs:cx*fzs+fzd,cy*fzs:cy*fzs+fzd]
          
          frequency_ratio.append(compute_frequency_ratio(subimage))
          entropy.append(compute_entropy(subimage))
          frequency_entropy.append(compute_frequency_entropy(subimage))
          
      
      plt.figure()
      plt.subplot(221)
      plt.imshow(gray_image.squeeze())
      plt.title("results image")
      plt.subplot(222)
      plt.imshow(np.array(frequency_ratio).reshape(freq_zoom_strides, freq_zoom_strides))
      plt.title("freq. ratio")
      plt.subplot(223)
      plt.imshow(np.array(entropy).reshape(freq_zoom_strides, freq_zoom_strides))
      plt.title("entropy")
      plt.subplot(224)
      plt.imshow(np.array(frequency_entropy).reshape(freq_zoom_strides, freq_zoom_strides))
      plt.title("frequency entropy")
      plt.tight_layout()
      plt.savefig(f"{root_dir}/assets/frequency_entropy_{time_stamp}_{idx}.png")
      #plt.show()
      
      params_list_nonblank =  np.array(params_list)[np.array(entropy) > 0]
      frequency_entropy_nonblank = np.array(frequency_entropy)[np.array(entropy) > 0]
      frequency_ratio_nonblank = np.array(frequency_ratio)[np.array(entropy) > 0]
      #params = params_list_nonblank[np.argmax(np.array(frequency_ratio_nonblank))]
      if np.sum(gray_image) == 0 or params_list_nonblank.shape[0] == 0:
          print("zoom no longer interesting, quitting")
          break
      params = params_list_nonblank[np.argmax(np.array(frequency_entropy_nonblank))]

      active_count = 0
      for ll in range(len(active_list)):
        if active_list[ll]:
          if active_count:
            if ll == 1:
              min_sigma = params[2]
              max_sigma = params[3]
            if ll == 2:
              min_dt = params[2]
              max_dt = params[3]
            if ll == 3: 
              min_kr = params[2]
              max_kr = params[3]
          else:
            if ll == 0:
              min_mu = params[0]
              max_mu = params[1]
            if ll == 1:
              min_sigma = params[0]
              max_sigma = params[1]
            if ll == 2:
              min_dt = params[0]
              max_dt = params[1]

          
          active_count += 1

        if active_count == 2:
          break

    t3 = time.time()
    idx += 1    
    time_elapsed = t3-t0
    total_zooms += 1
    

  for worker_idx in range(1, workers):
      print(f"send shutown signal to worker {worker_idx}")
      comm.send([-1], dest=worker_idx)

def arm(pattern, make_kernel, \
            dynamic_mode=0,\
            min_mu=0.15, max_mu=None, \
            min_sigma=0.017, max_sigma=None, \
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
            verbosity=0, \
            clipping_fn=lambda x: np.clip(x, 0.0, 1.0)):


  starting_grid = np.zeros((1, 1, grid_dim, grid_dim), dtype=default_dtype)

  while True:
    #comm.send((krs, dts, run_index), dest=worker_idx)
    my_input = comm.recv(source=0)

    if my_input[0] == -1:
        print(f"worker {rank} shutting down")
        break

    run_index = my_input[0]
    mus = my_input[1]
    sigmas = my_input[2]
    dts = my_input[3]
    krs = my_input[4]
    active_list = my_input[5]

    #print(f"run index {run_index} rec'd by worker {rank}")
    ii = int(np.floor(run_index / parameter_steps))
    jj = run_index % parameter_steps

    active_count = 0

    for ll in range(len(active_list)):
      
      if active_list[ll]:

        if ll == 0:
          mu = mus[ii]
        elif ll == 1:
          sigma = sigmas[ii] if active_count == 0 else sigmas[jj]
        elif ll == 2:
          dt = dts[ii] if active_count == 0 else dts[jj]
        elif ll == 3:
          kr = krs[jj]
          assert active_count == 1, "kr cannot be only active parameter in sweep"
            
        active_count += 1
      else:
        if ll == 0:
          mu = mus[0]
        elif ll == 1:
          sigma = sigmas[0]
        elif ll == 2:
          dt = dts[0]
        elif ll == 3:
          kr = krs[0]

    #print("active list", active_list)
    #print(f"worker {rank} starting {run_index} at {mus}, {sigmas}, {dts}, {krs}")
    #print(f"worker {rank} starting {run_index} at {mu}, {sigma}, {dt}, {kr}")
    kernel = make_kernel(kr.item())

    g_mode = 1 if dynamic_mode else 0
    my_update = make_update_function(mu, sigma, mode=g_mode)

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
    #print(f"worker {rank} finished {run_index} sim with acc. t: {accumulated_t_part.item()}")
    #print(f"dt: {dt.item():.3e} kr: {kr.item()}")
    #print(exploded, vanished, g)

    results_part = []

    results_part.append(run_index)
    results_part.append(accumulated_t_part)
    results_part.append(total_steps_counter)
    results_part.append(exploded)
    results_part.append(vanished)
    results_part.append(grid_0)
    results_part.append(grid)
    results_part.append(mu)
    results_part.append(sigma)
    results_part.append(dt)
    results_part.append(kr)

    if verbosity: print(f"worker {rank} sending results for {run_index} at {mu}, {sigma}, {dt}, {kr}")
    comm.send(results_part, dest=0)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("-n", "--pattern_name", default="asymdrop",\
      help="the name of the pattern to evaluate, e.g. orbium_unicaudatus. see patterns folder")
  parser.add_argument("-a", "--tag", default="exp",\
      help="a string tag to help you find your experiments")


  parser.add_argument("-p", "--parameter_steps", type=int, default=16,\
      help="number of parameter steps to sweep, result will be on a p by p grid")
  parser.add_argument("-r", "--max_runtime", type=float, default=1800,\
      help="(wall) time limit for experiment. experiments will always (try to) complete at least one run.")
  parser.add_argument("-t", "--max_t", type=float, default=16,\
      help="maximum accumulated simulation time (in time units)")
  parser.add_argument("-w", "--workers", type=int, default=16)

  parser.add_argument("-nmu", "--min_mu", type=float, default=0.15,\
      help="min value for mu (growth/target function peak). if no max_mu is provided, min_mu is used throughout")
  parser.add_argument("-xmu", "--max_mu", type=float, default=None,\
      help="max value for mu (growth/target function peak)")
  parser.add_argument("-ns", "--min_sigma", type=float, default=.017,\
      help="min value for sigma (growth/target function bell width). if no max_sigma is provided, min_mu is used throughout")
  parser.add_argument("-xs", "--max_sigma", type=float, default=None,\
      help="max value for sigma")

  parser.add_argument("-ndt", "--min_dt", type=float, default = 0.1,\
      help="min value for simulation time step dt. used throughout if max_dt is not provided.")
  parser.add_argument("-xdt", "--max_dt", type=float, default = None,\
      help="max value for step size dt")
  parser.add_argument("-nkr", "--min_kr", type=float, default = 13,\
      help="min value for kernel radius kr. can take float values. min_kr used throughout if max_kr not provided")
  parser.add_argument("-xkr", "--max_kr", type=float, default = None,\
      help="max value for kernel radius kr. can take float values.")
  parser.add_argument("-k0", "--k0", type=float, default=13,\
      help="native kr value, used to scale glider patterns. currently over-ridden by 13 for Orbium and 31 for Hydrogeminium patterns")

  parser.add_argument("-g", "--grid_dim", type=int, default = 256,\
      help="grid dimensions for simulation. this must be larger than 2*max_kr+3")
  parser.add_argument("-v", "--verbosity", type=int, default=0,\
      help="pass --verbosity 1 to print out debugging info during experiment.")

  args = parser.parse_args()

  pattern_name = args.pattern_name
  tag = args.tag
  workers = args.workers

  min_mu = args.min_mu
  max_mu = args.max_mu
  min_sigma = args.min_sigma
  max_sigma = args.max_sigma

  min_kr = args.min_kr
  max_kr = args.max_kr
  min_dt = args.min_dt
  max_dt = args.max_dt
  parameter_steps = args.parameter_steps 
  max_t = args.max_t 
  max_steps = max_t / args.min_dt 
  max_growth = 1.3 #args.max_growth 
  min_growth = 0.9 #args.min_growth 
  grid_dim = args.grid_dim 
  max_runtime = args.max_runtime

  stride = min([16, parameter_steps])
  default_dtype = np.float32
  kernel_dim = grid_dim - 6
  # k0 spec not actually used (overridden by the logic below)
  k0 = args.k0 
  verbosity = args.verbosity

  #### kernels
  if "orbium" or "asymdrop" or "scutium_gravidus" in pattern_name:
  
      #o. unicaudatus
      # the neighborhood kernel
      amplitudes = [1.0]
      means = [0.5]
      standard_deviations = [0.15]
      k0 = 13
  elif "hydrogeminium_natans" in pattern_name:
      # H. natans
      
      # the neighborhood kernel
      amplitudes = [0.5, 1.0, 0.6667]
      means = [0.0938, 0.2814, 0.4690]
      standard_deviations = [0.0330, 0.0330, 0.0330]
      k0 = 31


  root_dir = "."
  pattern_filepath = os.path.join(root_dir, "patterns", f"{pattern_name}.npy")

  pattern = np.load(pattern_filepath)

  make_kernel = make_make_kernel_function(amplitudes, means, \
          standard_deviations, dim=kernel_dim, default_dtype=default_dtype)

  dynamic_mode = 1 if "asym" in pattern_name else 0
  
  mpi_stability_sweep(pattern, make_kernel, dynamic_mode = dynamic_mode, \
        max_t = max_t, max_steps = max_steps, parameter_steps = parameter_steps, stride = stride,\
        grid_dim = grid_dim,\
        min_mu = min_mu, max_mu = max_mu,\
        min_sigma = min_sigma, max_sigma = max_sigma,\
        min_dt = min_dt, max_dt = max_dt,\
        min_kr = min_kr, max_kr = max_kr, k0 = k0, \
        default_dtype = default_dtype, \
        verbosity = verbosity, \
        workers = workers)

  print("finished")
