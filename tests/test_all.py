import unittest

from tests.functional_np.test_convolve import TestFTConvolveNp
from tests.functional_jax.test_convolve import TestFTConvolveJax
from tests.functional_jax.test_compose import TestMakeKernelField

from tests.functional_jax.test_metrics import TestComputeFrequencyRatio, \
    TestComputeEntropy,\
    TestComputeFrequencyEntropy

import fracatal

if __name__ == "__main__":

  unittest.main(verbosity=2)
