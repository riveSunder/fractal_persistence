import unittest
import os

import jax
import jax.numpy as np
import numpy.random as npr

from fracatal.functional_jax.compose import make_gaussian, \
    make_mixed_gaussian, \
    make_kernel_field, \
    make_update_function, \
    make_update_step, \
    make_make_kernel_function, \
    sigmoid_1, \
    make_smooth_steps_function, \
    make_make_smoothlife_kernel_function, \
    make_smooth_interval, \
    make_smoothlife_update_function, \
    make_smoothlife_update_step


class TestMakeKernelField(unittest.TestCase):

  def setUp(self):
    pass

  def test_make_kernel_field(self):

    dim = 126
    kr = dim // 2

    jax.config.update("jax_enable_x64", True)
    for my_dtype in [np.float16, np.float32, np.float64]:

      kernel_field = make_kernel_field(kr, dim=dim, default_dtype=my_dtype) 

      self.assertEqual(kernel_field.dtype, my_dtype)

      self.assertEqual(dim+1, kernel_field.shape[-2])
      self.assertEqual(dim+1, kernel_field.shape[-1])

      self.assertEqual(2, len(kernel_field.shape))


    

