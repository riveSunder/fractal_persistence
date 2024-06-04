import numpy.random as npr
import numpy as np

import torch

def pad_2d(kernel, pad_to):
  """
  pads the last two dimensions of grid to match dims of pad_to
  """

  if kernel.shape[1:] != pad_to[1:]:

    diff_h  = pad_to[-2] - kernel.shape[-2]
    diff_w =  pad_to[-1] - kernel.shape[-1]
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

    padded_kernel = torch.nn.functional.pad(kernel, \
        (pad_w+wp, pad_w+wm, pad_h+hp, pad_h+hm, 0,0, 0,0))
  else:
    return kernel

  return padded_kernel

