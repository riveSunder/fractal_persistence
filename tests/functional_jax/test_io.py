import unittest


from fracatal.functional_jax.io import load_rule_dict,\
    load_rule

class TestLoadRuleDict(unittest.TestCase):

  def setUp(self):
    pass

  def test_load_rule_dict(self):

    rule_dict = load_rule_dict()

    self.assertEqual(type(rule_dict), type({}))


class TestLoadRule(unittest.TestCase):

  def setUp(self):
    pass

  def test_load_rule(self):

    rule_dict = load_rule_dict()

    for my_key in rule_dict.keys():

      rule = load_rule(name=my_key)

      self.assertEqual(type(rule), type({}))

if __name__ == "__main__":

  unittest.main(verbosity=2)
