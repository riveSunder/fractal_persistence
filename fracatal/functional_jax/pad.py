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

