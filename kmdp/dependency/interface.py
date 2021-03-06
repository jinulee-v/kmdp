"""
@module interface
provide interfaces to access rules
"""

from . import _kmdp_rule_dict, _kmdp_precedence
from .rules_base import KMDPRuleBase

def is_local(alias):
  if issubclass(_kmdp_rule_dict[alias], KMDPRuleBase):
    return True
  return False


def kmdp_generate(alias, *args):
  if is_local(alias):
    return kmdp_generate_local(alias, *args)
  else:
    return kmdp_generate_global(alias, *args)

def kmdp_generate_global(alias, sentence, dep_id, dep_wp_i):
  return _kmdp_rule_dict[alias].generate(sentence, dep_id, dep_wp_i)

def kmdp_generate_local(alias, dep_wp, dep_wp_i, head_wp, dp_label):
  return _kmdp_rule_dict[alias].generate(dep_wp, dep_wp_i, head_wp, dp_label)

def kmdp_recover(alias, sentence, dep_id, dep_wp_i):
  if issubclass(_kmdp_rule_dict[alias], KMDPRuleBase):
    dep_wp = sentence['pos'][dep_id]
    head_wp = sentence['pos'][sentence['dependency'][dep_id]['head']-1]
    dp_label = sentence['dependency'][dep_id]['label']
    return _kmdp_rule_dict[alias].recover(dep_wp, dep_wp_i, head_wp, dp_label)
  return _kmdp_rule_dict[alias].recover(sentence, dep_id, dep_wp_i)


def get_doc(alias):
  """
  Returns docstring of the rule.
  """
  if alias in _kmdp_rule_dict:
    return _kmdp_rule_dict[alias].__doc__
  raise ValueError('Rule {} does not exist'.format(alias))

def find_dominated_rules(alias):
  """
  Returns all successor(overrided) rules of a given rule.
  Performs DFS search within the tree.
  """
  rules = [alias]
  for dominee in _kmdp_precedence[alias]:
    rules.extend(find_dominated_rules(dominee))
  return rules

class KMDPGenerateException(Exception):
  """
  @class KMDPGenerateException

  Exception class when generate() produces error(unexpected behavior).
  """
  def __init__(self, rule, msg, dep_wp, dep_wp_i, head_wp, dp_label):

    super(KMDPGenerateException, self).__init__(msg)

    self.rule = rule
    self.dep_wp = dep_wp
    self.dep_wp_i = dep_wp_i
    self.head_wp = head_wp
    self.dp_label = dp_label
    self.environment = {
      'rule': rule,
      'msg': msg,
      'dep_wp': dep_wp,
      'dep_wp_i': dep_wp_i,
      'head_wp': head_wp,
      'dp_label': dp_label
    }
  

class KMDPRecoverException(Exception):
  """
  @class KMDPGenerateException

  Exception class when recover() produces error(unexpected behavior).
  """
  def __init__(self, name, msg, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):

    super(KMDPRecoverException, self).__init__(msg)

    self.rule = rule
    self.dep_wp = dep_wp
    self.dep_wp_i = dep_wp_i
    self.head_wp = head_wp
    self.head_wp_i = head_wp_i
    self.kmdp_label = kmdp_label
    self.environment = {
      'rule': rule,
      'msg': msg,
      'dep_wp': dep_wp,
      'dep_wp_i': dep_wp_i,
      'head_wp': head_wp,
      'head_wp_i': head_wp_i,
      'kmdp_label': kmdp_label
    }
  