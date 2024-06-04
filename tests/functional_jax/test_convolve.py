import unittest

import jax.numpy as np
import numpy.random as npr

from fracatal.functional_jax.convolve import ft_convolve


class TestFTConvolveJax(unittest.TestCase):

  def setUp(self):
    npr.seed(13)

  def test_ft_convolve_id(self):

    id_kernel = np.zeros((1,1,3,3))
    id_kernel = id_kernel.at[:,:,1,1].set(1.0)

    empty_kernel = np.zeros((1,1,3,3))

    for dim in [8, 16, 32]:
      grid_0 = 1.0 * (npr.rand(1,1,32,32) > 0.5)

      grid_1 = ft_convolve(grid_0, id_kernel)
      grid_2 = ft_convolve(grid_0, empty_kernel)

      # assert equal to about 6 digits (32-bit floats give us about ~7 digits of precision)
      self.assertAlmostEqual(0.0, np.abs(grid_1 - grid_0).mean(), places=6)
      self.assertAlmostEqual(0.0, grid_2.mean(), places=6)
      

  def test_ft_convolve_zero(self):

    empty_kernel = np.zeros((1,1,3,3))

    for dim in [8, 16, 32]:
      grid_0 = 1.0 * (npr.rand(1,1,32,32) > 0.5)

      grid_2 = ft_convolve(grid_0, empty_kernel)
      self.assertAlmostEqual(0.0, grid_2.mean(), places=6)
      
if __name__ == "__main__":

  unittest.main(verbosity=2)
