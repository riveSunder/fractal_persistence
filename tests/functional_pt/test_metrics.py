import unittest

import numpy as np
import torch

import numpy.random as npr

from fracatal.functional_pt.metrics import compute_entropy, \
    compute_frequency_ratio, \
    compute_frequency_entropy


class TestComputeFrequencyRatioPT(unittest.TestCase):

  def setUp(self):
    pass

  def test_compute_frequency_ratio(self):

    a = torch.rand(1,1,32,32)
    b = torch.ones((1,1,32,32))

    frequency_ratio_a = compute_frequency_ratio(a)
    frequency_ratio_b = compute_frequency_ratio(b)

    self.assertLess(frequency_ratio_b, frequency_ratio_a)

class TestComputeEntropyPT(unittest.TestCase):

  def setUp(self):
    pass

  def test_compute_entropy(self):

    a = torch.rand(1,1,32,32)
    b = torch.ones((1,1,32,32))

    frequency_ratio_a = compute_entropy(a)
    frequency_ratio_b = compute_entropy(b)

    self.assertLess(frequency_ratio_b, frequency_ratio_a)

class TestComputeFrequencyEntropyPT(unittest.TestCase):

  def setUp(self):
    pass

  def test_compute_frequency_entropy(self):

    a = torch.rand(1,1,32,32)
    b = torch.ones((1,1,32,32))

    frequency_ratio_a = compute_frequency_entropy(a)
    frequency_ratio_b = compute_frequency_entropy(b)

    self.assertLess(frequency_ratio_b, frequency_ratio_a)

if __name__ == "__main__":

  unittest.main(verbosity=2)
