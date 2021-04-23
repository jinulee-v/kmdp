"""
@module interface
provide interfaces to access rules
"""

from . import _kmdp_rule_dict

def kmdp_generate(alias, *args):
  return _kmdp_rule_dict[alias].generate(*args)

def kmdp_recover(alias, *args):
  return _kmdp_rule_dict[alias].recover(*args)