"""
@module dependency
Manage registered dependency rules.
"""

# Decorator and data structures for management
# Invisible to external modules.
_kmdp_rule_dict = {}
_kmdp_precedence = {}

def register_kmdp_rule(alias, dominees):
  """
  Decorator to manage rule functions.

  @param cls KMDP rule class.
  @param alias An alias for `fn`. This alias should be given to call the rule class.
  @param dominees alias for other rule class that `cls` dominates.
  """
  def decorator(cls):
    _kmdp_rule_dict[alias] = cls
    _kmdp_precedence[alias] = dominees
  return decorator

# Import and organize rule_based dependency functions.
# All rule functions should use @register_kmdp_rule decorator to be properly accessed.

from .rules import *

# In this point, decorator is applied

from .manage_rules import topological_sort

kmdp_rules = topological_sort(_kmdp_precedence)

from .interface import kmdp_generate, kmdp_recover