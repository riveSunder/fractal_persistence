import sys
import argparse
import subprocess
import os

import numpy.random as npr
import numpy as np
import torch

import time

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
matplotlib.rcParams["animation.embed_limit"] = 1024

import skimage
import skimage.io as sio
import skimage.transform
import fracatal

from fracatal.functional_pt.convolve import ft_convolve
from fracatal.functional_pt.pad import pad_2d
from fracatal.functional_pt.metrics import compute_entropy, \
        compute_frequency_ratio, \
        compute_frequency_entropy
from fracatal.functional_pt.compose import make_gaussian, \
        make_mixed_gaussian, \
        make_kernel_field, \
        make_update_function, \
        make_update_step, \
        make_make_kernel_function

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
            default_dtype=torch.float32, \
            clipping_fn=lambda x: torch.clamp(x, 0.0, 1.0), \
            verbosity=0,\
            workers=8,
            device=torch.device("cpu")):

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
            clipping_fn, workers, \
            device=device)
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
            clipping_fn, \
            device=device)


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
            default_dtype=torch.float32, \
            verbosity=0, \
            clipping_fn=lambda x: torch.clamp(x, 0.0, 1.0),\
            workers=1,
            device=torch.device("cpu")):

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

    # new priority
    # mu > dt > sigma > kr 
    # hierarchy of parameters mu > sigma > dt > kr
    # in other words, if max_mu and max_sigma are both specified, dt and kr are static

    if max_mu is not None:
      max_mu, min_mu = max([max_mu, min_mu]), min([max_mu, min_mu])
      mus = torch.arange(max_mu, min_mu, -(max_mu-min_mu) / parameter_steps, dtype=default_dtype)[:,None, None,None]
      active_mu = True
    else:
      mus = torch.tensor([min_mu])[:,None,None,None]
      active_mu = False


    if max_dt is not None:
      max_dt, min_dt = max([max_dt, min_dt]), min([max_dt, min_dt])
      if max_mu is None:
        dts = torch.arange(max_dt, min_dt, -(max_dt-min_dt) / parameter_steps, dtype=default_dtype)[:,None, None,None]
      else: 
        dts = torch.arange(min_dt, max_dt, (max_dt-min_dt) / parameter_steps, dtype=default_dtype)[:,None, None,None]
      active_dt = True
    else:
      dts = torch.tensor([min_dt])[:,None,None,None]
      active_dt = False

    if (not(active_mu and active_dt)) and max_sigma is not None:
      max_sigma, min_sigma = max([max_sigma, min_sigma]), min([max_sigma, min_sigma])
      sigmas = torch.arange(min_sigma, max_sigma, (max_sigma-min_sigma) / parameter_steps, dtype=default_dtype)[:,None, None,None]
      active_sigma = True
    else:
      sigmas = torch.tensor([min_sigma])[:,None,None,None]
      active_sigma = False
      
    if (not(active_mu or active_dt) or not(active_sigma)) and max_kr is not None:
      max_kr, min_kr = max([max_kr, min_kr]), min([max_kr, min_kr])
      krs = torch.arange(min_kr, max_kr, (max_kr-min_kr) / parameter_steps, dtype=default_dtype)[:,None,None,None]
      active_kr = True
    else:
      krs = torch.tensor([min_kr])[:,None,None,None]
      active_kr = False

    mus = mus.to(device)
    dts = dts.to(device)
    sigmas = sigmas.to(device)
    krs = krs.to(device)

    active_list = [active_mu, active_dt, active_sigma, active_kr]
    assert torch.sum(torch.tensor([active_kr, active_dt, active_sigma, active_mu])) == 2, "only two parameters should have ranges"
      
    results_img = np.zeros((parameter_steps, parameter_steps, 4))

    native_dim_h = pattern.shape[-2]
    native_dim_w = pattern.shape[-1]

    explode = torch.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    vanish = torch.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    done = torch.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    accumulated_t = torch.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    total_steps = torch.zeros((parameter_steps, parameter_steps,1,1), dtype=default_dtype)
    starting_grid = torch.zeros((1, 1, grid_dim, grid_dim), dtype=default_dtype)

    red_cmap = plt.get_cmap("Reds")
    green_cmap = plt.get_cmap("Greens")
    blue_cmap = plt.get_cmap("Blues")

    run_index = 0
    run_max = parameter_steps**2 

    while run_index < run_max:
        ## parallel mpi section ****
        last_worker = 1 * workers
        for worker_index in range(1, workers):
          if run_index < run_max:
            comm.send((run_index, mus, sigmas, dts, krs, active_list), dest=worker_index)
            if verbosity: print(f"run index {run_index} sent to worker {worker_index}")
            run_index += 1
          else:
            last_worker = worker_index
            break

          

        #for worker_idx in range(1, last_worker):
        total_returned = 0
        while total_returned < (run_max):

          #if verbosity: print(f"rec'ing from worker {worker_idx} of {last_worker-1}")

          if total_returned < run_max:
            results_part = comm.recv()
            worker_index = results_part[11]
            run_index_part = results_part[0]
            if verbosity: print(f"rec'ed {run_index_part} from worker {worker_index}")

            #max_run_index_returned = max([run_index_part, max_run_index_returned])
            total_returned += 1

          if verbosity: print(f"received {total_returned} so far")
          if run_index < run_max:
            comm.send((run_index, mus, sigmas, dts, krs, active_list), dest=worker_index)
            if verbosity: print(f"run index {run_index} to worker {worker_index}")
            run_index += 1

          
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
                  y_ticks = 1.0 * dts #sigmas
                  ylabel = "$\Delta t$" #"$\sigma$"
                  min_y = min_dt #sigma
                  max_y = max_dt #sigma
                elif ll == 2:
                  y_ticks = 1.0 * sigmas #dts
                  ylabel = "$\sigma$" #Delta t$"
                  min_y = min_sigma #dt
                  max_y = max_sigma #dt
              elif count_active == 2:
                if ll == 1:
                  x_ticks = 1.0 * dts #sigmas
                  xlabel = "$\Delta t$" #sigma$"
                  min_x = min_dt #sigma
                  max_x = max_dt #sigma
                elif ll == 2:
                  x_ticks = 1.0 * sigmas #dts
                  xlabel = "$\sigma$" #Delta t$"
                  min_x = min_sigma #dt
                  max_x = max_sigma #dt
                elif ll == 3:
                  x_ticks = 1.0 * krs
                  xlabel = "$k_r$"
                  min_x = min_kr
                  max_x = max_kr
                break

          if verbosity: print(ii, jj, param_indices)
          mu_check = mus[param_indices[0]]
          dt_check = dts[param_indices[1]]
          sigma_check = sigmas[param_indices[2]]
          kr_check = krs[param_indices[3]]

          if not(torch.isclose(mu_check, mu)): 
            print("mu rec'd from worker does not match")
            print(f"{mu_check}, {mu}, {worker_index}")
          if not(torch.isclose(sigma_check, sigma)):
            print("sigma rec'd from worker does not match")
            print(f"{sigma_check}, {sigma}, {worker_index}")
          if not(torch.isclose(dt_check, dt)):
            print("dt rec'd from worker does not match")
            print(f"{dt_check}, {dt}, {worker_index}")
          if not(torch.isclose(kr_check, kr)):
            print("kr rec'd from worker does not match")
            print(f"{kr_check}, {kr}, {worker_index}")

          assert mu == mu_check, f"expected mu={mu_check}, got {mu} from worker {worker_idx}/{run_index_part}"
          assert sigma == sigma_check,  f"expected sigma={sigma_check}, got {sigma} from worker {worker_idx}/{run_index_part}"
          assert dt == dt_check,  f"expected mu={dt_check}, got {dt} from worker {worker_idx}/{run_index_part}"
          assert kr == kr_check,  f"expected mu={kr_check}, got {kr} from worker {worker_idx}/{run_index_part}"

          accumulated_t_part = accumulated_t_part
          accumulated_t_truncated = torch.clamp(accumulated_t_part, 0, max_t).cpu()

          if exploded == True:
            results_img[ii,jj] = red_cmap(accumulated_t_truncated / max_t)
          elif vanished == True:
            results_img[ii,jj] = blue_cmap(accumulated_t_truncated / max_t)
          else:
            results_img[ii,jj] = green_cmap(accumulated_t_truncated / max_t)

          accumulated_t[ii,jj] = accumulated_t_part
          total_steps[ii,jj] = total_steps_part
          explode[ii,jj] = exploded
          vanish[ii,jj] = vanished
          done[ii,jj] = exploded or vanished


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
    msg = f"    {xlabel} from {min_x:.8e} to {max_x:.8e}\n"
    msg += f"   {ylabel} from {min_y:.8e} to {max_y:.8e}\n"
    
    ax.set_title("disco persistence \n" + msg, fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{root_dir}/assets/disco_{exp_name}_{idx}.png")
       
    print(msg2 + msg)
    # save results
    
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
    np.save(grid_0_savepath, results[-1][6].cpu())
    np.save(grid_T_savepath, results[-1][7].cpu())
    
    # log experiment metadata
    if metadata_path is None:
      metadata_path = os.path.join(save_dir, f"metadata_{time_stamp}.txt")
      metadata = "index, pattern_name, min_mu, max_mu, min_dt,  max_dt, min_sigma, max_sigma, "
      metadata += "min_kr, max_kr, parameter_steps, max_t, max_steps, max_runtime, "
      metadata += "time_stamp, sim_time_elapsed,  total_time_elapsed, "
      metadata += "img_savepath, accumulated_t_savepath, total_steps_savepath, explode_savepath, vanish_savepath, grid_T_savepath\n"

      with open(metadata_path,"w") as f:
          f.write(metadata)

    metadata = f"{idx}, {pattern_name}, {mus.min().item()}, {mus.max().item()}, {dts.min().item()}, "
    metadata += f"{dts.max().item()}, "
    metadata += f"{sigmas.min().item()}, {sigmas.max().item()}, {krs.min().item()}, "
    metadata += f"{krs.max().item()}, {parameter_steps}, {max_t}, {max_steps}, {max_runtime}, "
    metadata += f"{time_stamp}, {t2-t1:2f}, {t2-t0:2f}, "
    metadata += f"{img_savepath}, {accumulated_t_savepath}, {total_steps_savepath}, {explode_savepath}, {vanish_savepath}, {grid_T_savepath}\n"
    with open(metadata_path,"a") as f:
        f.write(metadata)
    # determine next parameter range
    if (time.time()-t0) < max_runtime:
      freq_zoom_dim = (parameter_steps) // freq_zoom_fraction
      freq_zoom_dim = min([freq_zoom_dim, parameter_steps // 2])
      freq_zoom_stride = freq_zoom_dim // 2 #4 + int(parameter_steps/4)
      freq_zoom_strides = (parameter_steps-freq_zoom_dim) // freq_zoom_stride +1
      
      fzd = freq_zoom_dim
      fzs = freq_zoom_stride
      
      params_list = []
      entropy = []
      frequency_entropy = []
      frequency_ratio = []
      # Weighted RGB conversion to grayscale
      gray_image = (1.0 - results[-1][5])

      for ch in range(0, parameter_steps-fzd+1, fzs): 
          for cw in range(0, parameter_steps-fzd+1, fzs):
            params_list.append([y_ticks[ch].item(), \
                    y_ticks[ch+fzd-1].item(),\
                    x_ticks[cw].item(), \
                    x_ticks[cw+fzd-1].item()])


            resize_to = max([parameter_steps // 16, 32])
            subimage = gray_image[ch:ch+fzd,cw:cw+fzd]
            subimage = torch.tensor(skimage.transform.resize(subimage.cpu().numpy(), \
                (resize_to, resize_to, 1, 1), order=5, anti_aliasing=True))

            #frequency_ratio.append(compute_frequency_ratio(subimage))
            #entropy.append(compute_entropy(subimage))
            frequency_entropy.append(compute_frequency_entropy(subimage))
            if verbosity: print(ch, y_ticks[ch], cw, x_ticks[cw], frequency_entropy[-1])

      plt.figure()
      plt.subplot(121)
      plt.imshow(gray_image.squeeze())
      plt.title("results image")
      plt.subplot(122)
      plt.imshow(np.array(frequency_entropy).reshape(freq_zoom_strides, freq_zoom_strides))
      plt.title("frequency entropy")
      plt.tight_layout()
      plt.savefig(f"{root_dir}/assets/frequency_entropy_{time_stamp}_{idx}.png")

      params_list_nonblank =  torch.tensor(params_list)[torch.tensor(frequency_entropy) > 0]
      frequency_entropy_nonblank = torch.tensor(frequency_entropy)[torch.tensor(frequency_entropy) > 0]
      #frequency_ratio_nonblank = torch.tensor(frequency_ratio)[torch.tensor(entropy) > 0]

      if torch.sum(gray_image) == 0 or params_list_nonblank.shape[0] == 0:
          print("zoom no longer interesting, quitting")
          break

      params = params_list_nonblank[np.argmax(np.array(frequency_entropy_nonblank))]

      active_count = 0
      for ll in range(len(active_list)):
        if active_list[ll]:
          if active_count:
            if ll == 1:
              if active_mu:
                min_dt = params[2].to(device)
                max_dt = params[3].to(device)
              else:
                max_dt = params[2].to(device)
                min_dt = params[3].to(device)
            if ll == 2:
              min_sigma = params[2].to(device)
              max_sigma = params[3].to(device)
            if ll == 3: 
              min_kr = params[2].to(device)
              max_kr = params[3].to(device)
          else:
            if ll == 0:
              min_mu = params[0].to(device)
              max_mu = params[1].to(device)
            if ll == 1:
              if active_mu:
                min_dt = params[0].to(device)
                max_dt = params[1].to(device)
              else:
                max_dt = params[0].to(device)
                min_dt = params[1].to(device)

            if ll == 2:
              min_sigma = params[0].to(device)
              max_sigma = params[1].to(device)

          
          active_count += 1

        if active_count == 2:
          break
    t3 = time.time()

    idx += 1    
    time_elapsed = t3-t0
    total_zooms += 1
    

  for worker_idx in range(1, workers):
      if verbosity: print(f"send shutown signal to worker {worker_idx}")
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
            default_dtype=torch.float32, \
            verbosity=0, \
            clipping_fn=lambda x: torch.clamp(x, 0.0, 1.0), \
            device=torch.device("cpu")):


  starting_grid = torch.zeros((1, pattern.shape[1], grid_dim, grid_dim), dtype=default_dtype)

  while True:
    #comm.send((krs, dts, run_index), dest=worker_idx)
    my_input = comm.recv(source=0)

    if my_input[0] == -1:
        if verbosity: print(f"worker {rank} shutting down")
        break

    run_index = my_input[0]
    mus = my_input[1].to(device)
    sigmas = my_input[2].to(device)
    dts = my_input[3].to(device)
    krs = my_input[4].to(device)
    active_list = my_input[5]

    if verbosity: print(f"run index {run_index} rec'd by worker {rank}")
    ii = int(np.floor(run_index / parameter_steps))
    jj = run_index % parameter_steps
    if verbosity: print(f"p {parameter_steps}, {ii}, {jj}")

    active_count = 0

    for ll in range(len(active_list)):
      
      if active_list[ll]:

        if ll == 0:
          mu = mus[ii]
        elif ll == 1:
          dt = dts[ii] if active_count == 0 else dts[jj]
        elif ll == 2:
          sigma = sigmas[ii] if active_count == 0 else sigmas[jj]
        elif ll == 3:
          kr = krs[jj]
          assert active_count == 1, "kr cannot be only active parameter in sweep"
            
        active_count += 1
      else:
        if ll == 0:
          mu = mus[0]
        elif ll == 1:
          dt = dts[0]
        elif ll == 2:
          sigma = sigmas[0]
        elif ll == 3:
          kr = krs[0]

    kernel = make_kernel(kr.item()).to(device)

    g_mode = dynamic_mode #1 if dynamic_mode else 0
    my_update = make_update_function(mu, sigma, mode=g_mode, device=device)

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
      scaled_pattern = torch.tensor(skimage.transform.rescale(pattern, (1,1, scale_factor, scale_factor), order=5, anti_aliasing=True), \
        dtype=default_dtype)
    else:
      scaled_pattern = torch.tensor(skimage.transform.rescale(pattern, (1,1, scale_factor, scale_factor), order=5), dtype=default_dtype)

    starting_grid[:,:,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]] = scaled_pattern

    grid = starting_grid * 0.0
    grid[:,:,:scaled_pattern.shape[-2], :scaled_pattern.shape[-1]] = scaled_pattern
    grid = grid.to(device)

    kernel = pad_2d(kernel, grid.shape)

    starting_sum = grid[:,0,:,:].sum()

    exploded = False
    vanished = False
    total_steps_counter = 0
    accumulated_t_part = 0.0

    grid_0 = 1.0 * grid
    while accumulated_t_part < max_t and total_steps_counter <= max_steps:

      grid = update_step(grid)

      g = grid[:,0,:,:].sum() / starting_sum

      accumulated_t_part += dt
      total_steps_counter += 1

      if g > max_growth:
        exploded = True
        break
      if g < min_growth:
        vanished = True
        break
    #print(min_growth, max_growth, g, exploded, vanished)
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
    results_part.append(rank)

    if verbosity: print(f"worker {rank} sending results for {run_index} at {mu}, {sigma}, {dt}, {kr}")
    comm.send(results_part, dest=0)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("-a", "--tag", default="exp",\
      help="a string tag to help you find your experiments")
  parser.add_argument("-n", "--pattern_name", default="asymdrop",\
      help="the name of the pattern to evaluate, e.g. orbium_unicaudatus. see patterns folder")

  parser.add_argument("-m", "--my_device", type=str, default="cpu",\
      help="device for torch to use")

  parser.add_argument("-p", "--parameter_steps", type=int, default=16,\
      help="number of parameter steps to sweep, result will be on a p by p grid")
  parser.add_argument("-r", "--max_runtime", type=float, default=1800,\
      help="(wall) time limit for experiment. experiments will always (try to) complete at least one run.")
  parser.add_argument("-t", "--max_t", type=float, default=16,\
      help="maximum accumulated simulation time (in time units)")
  parser.add_argument("-w", "--workers", type=int, default=16)

  parser.add_argument("-ng", "--min_growth", type=float, default=.9,\
      help="minimum relative mass for determining mass homeostasis")
  parser.add_argument("-xg", "--max_growth", type=float, default=1.3,\
      help="maximum relative mass for determining mass homeostasis")

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
  max_growth = args.max_growth 
  min_growth = args.min_growth 
  grid_dim = args.grid_dim 
  max_runtime = args.max_runtime
  my_device = torch.device(args.my_device)

  stride = min([16, parameter_steps])
  default_dtype = torch.float32
  kernel_dim = grid_dim - 6
  # k0 spec not actually used (overridden by the logic below)
  k0 = args.k0 
  verbosity = args.verbosity

  #### kernels
  if "orbium" in pattern_name or \
      "asymdrop" in pattern_name or \
      "scutium_gravidus" in pattern_name or\
      "gyropteron":
  
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
          standard_deviations, dim=kernel_dim, default_dtype=default_dtype, device=my_device)

  if "asymdrop" == pattern_name:
    dynamic_mode = 1
  elif "adorbium" == pattern_name:
    dynamic_mode = 3
  else:
    dynamic_mode = 0
  
  with torch.no_grad():
    mpi_stability_sweep(pattern, make_kernel, dynamic_mode = dynamic_mode, \
          max_t = max_t, max_steps = max_steps, parameter_steps = parameter_steps, stride = stride,\
          grid_dim = grid_dim,\
          min_growth=min_growth, max_growth=max_growth, \
          min_mu = min_mu, max_mu = max_mu,\
          min_sigma = min_sigma, max_sigma = max_sigma,\
          min_dt = min_dt, max_dt = max_dt,\
          min_kr = min_kr, max_kr = max_kr, k0 = k0, \
          default_dtype = default_dtype, \
          verbosity = verbosity, \
          workers = workers, 
          device = my_device)

  if verbosity: print("finished")
