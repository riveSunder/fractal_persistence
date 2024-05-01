import jax
from jax import numpy as np
import numpy.random as npr

import os
import json

import skimage
import skimage.io as sio


THIS_FILE_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.split(os.path.split(os.path.split(THIS_FILE_PATH)[0])[0])[0]
RULE_PATH = os.path.join(ROOT_DIR, "patterns", "rules.json")

def load_rule_dict():

  with open(RULE_PATH, "r") as f:
    rule_dict = json.load(f)

  return rule_dict

def load_rule(name):

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

def save_rule_dict(rule_dict):

  with open(RULE_PATH, "w") as f:
    json.dump(rule_dict, f)
  

def add_rule(rule):

  rule_dict = load_rule_dict()
  rule_name = rule.keys()[0]

  if rule_name in rule_dict.keys():
    print(f"rule with name {rule_name} already in rules")

    print("rule names: ")

    for my_key in rule_dict.keys():
      print(f"\t\t {my_key}")

    return -1

  else:
    rule_dict[rule_name] = rule[rule_name]

  save_rule_dict(rule_dict)


def load_pattern_and_rule(name, load_image=False):

  rule = load_rule(name)
  if rule == -1:
    print(f"no rule set found for {name}")
    return -1

  if load_image:
    pattern = sio.imread(os.path.join(ROOT_DIR, "patterns", f"{name}.png"))
    pattern = np.array(pattern)[None,None,:,:]
  else:
    pattern = np.load(os.path.join(ROOT_DIR, "patterns", f"{name}.npy"))

  return pattern, rule
