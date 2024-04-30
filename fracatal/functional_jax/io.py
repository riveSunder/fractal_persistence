import jax
from jax import numpy as np
import numpy.random as npr

import os
import json


THIS_FILE_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.split(os.path.split(os.path.split(THIS_FILE_PATH)[0])[0])[0]
RULE_PATH = os.path.join(ROOT_DIR, "patterns", "rules.json")

def load_rule_dict():

  with open(RULE_PATH, "r") as f:
    rule_dict = json.load(f)

  return rule_dict

def load_rule(name="orbium_unicaudatus"):

  rule_dict = load_rule_dict()

  if name in rule_dict.keys():
    rule_result = rule_dict[name]
  else:
    msg = f"Name {name} not found in rules\n\t available rule keys are:"
    print(msg)
    for my_key in rule_dict.keys():
      print(f"\t\t {my_key}")

    return -1
  
  return rule_result
