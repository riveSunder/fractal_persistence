import unittest

from tests.functional_np.test_convolve import TestFTConvolveNp
from tests.functional_jax.test_convolve import TestFTConvolveJax
from tests.functional_jax.test_compose import TestMakeKernelField
from tests.functional_jax.test_io import TestLoadRuleDict,\
    TestLoadRule

from tests.functional_jax.test_metrics import TestComputeFrequencyRatio, \
    TestComputeEntropy,\
    TestComputeFrequencyEntropy

import fracatal
from fracatal.scripts.mpi_sweep import *
from fracatal.scripts.v_stability_sweep import *
from fracatal.scripts.stability_sweep import *

if __name__ == "__main__":

  unittest.main(verbosity=2)
