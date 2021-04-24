"""
@module interface
provide interfaces to access rules
"""

from . import _kmdp_rule_dict

def kmdp_generate(alias, dep_wp, dep_wp_i, head_wp, dp_label):
  return _kmdp_rule_dict[alias].generate(*args)

def kmdp_recover(alias, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):
  return _kmdp_rule_dict[alias].recover(*args)


class KMDPGenerateException(Exception):
  """
  @class KMDPGenerateException

  Exception class when generate() produces error(unexpected behavior).
  """
  def __init__(rule, msg, dep_wp, dep_wp_i, head_wp, dp_label):

    super(KMDPGenerateException, self).__init__(msg)

    self.rule = rule
    self.dep_wp = dep_wp
    self.dep_wp_i = dep_wp_i
    self.head_wp = head_wp
    self.dp_label = dp_label
    self.environment = {
      'rule': rule,
      'dep_wp': dep_wp
      'dep_wp_i': dep_wp_i
      'head_wp': head_wp
      'dp_label': dp_label
    }
  

class KMDPRecoverException(Exception):
  """
  @class KMDPGenerateException

  Exception class when recover() produces error(unexpected behavior).
  """
  def __init__(cls, msg, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):

    super(KMDPRecoverException, self).__init__(msg)

    self.rule = rule
    self.dep_wp = dep_wp
    self.dep_wp_i = dep_wp_i
    self.head_wp = head_wp
    self.head_wp_i = head_wp_i
    self.kmdp_label = kmdp_label
    self.environment = {
      'rule': rule,
      'dep_wp': dep_wp
      'dep_wp_i': dep_wp_i
      'head_wp': head_wp
      'head_wp_i': head_wp_i
      'kmdp_label': kmdp_label
    }
  