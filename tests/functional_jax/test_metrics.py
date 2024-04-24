import unittest

import jax.numpy as np
import numpy.random as npr

from fracatal.functional_jax.metrics import compute_entropy, \
    compute_frequency_ratio, \
    compute_frequency_entropy


class TestComputeFrequencyRatio(unittest.TestCase):

  def setUp(self):
    pass

  def test_compute_frequency_ratio(self):

    a = npr.rand(1,1,32,32)
    b = np.ones((1,1,32,32))

    frequency_ratio_a = compute_frequency_ratio(a)
    frequency_ratio_b = compute_frequency_ratio(b)

    self.assertLess(frequency_ratio_b, frequency_ratio_a)

class TestComputeEntropy(unittest.TestCase):

  def setUp(self):
    pass

  def test_compute_entropy(self):

    a = npr.rand(1,1,32,32)
    b = np.ones((1,1,32,32))

    frequency_ratio_a = compute_frequency_ratio(a)
    frequency_ratio_b = compute_frequency_ratio(b)

    self.assertLess(frequency_ratio_b, frequency_ratio_a)

class TestComputeFrequencyEntropy(unittest.TestCase):

  def setUp(self):
    pass

  def test_compute_frequency_entropy(self):

    a = npr.rand(1,1,32,32)
    b = np.ones((1,1,32,32))

    frequency_ratio_a = compute_frequency_ratio(a)
    frequency_ratio_b = compute_frequency_ratio(b)

    self.assertLess(frequency_ratio_b, frequency_ratio_a)
