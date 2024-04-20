import unittest

from tests.functional_np.test_convolve import TestFTConvolveNp
from tests.functional_jax.test_convolve import TestFTConvolveJax

import fracatal

if __name__ == "__main__":

  unittest.main(verbosity=2)
