"""
@module dependency
Manage registered dependency rules.
"""

# Decorator and data structures for management
# Invisible to external modules.
_kmdp_rule_dict = {}
_kmdp_precedence = {}
kmdp_rules = None

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

def register_global_rule(alias, dominees):
  """
  Decorator to manage global rule functions.

  @param cls global KMDP rule class.
  @param alias An alias for `fn`. This alias should be given to call the rule class.
  @param dominees alias for other rule class that `cls` dominates.
         c.f. All global rules is hard-coded to precede local rules(in rules.py).
  """
  def decorator(cls):
    _kmdp_rule_dict[alias] = cls
    _kmdp_precedence[alias] = dominees + kmdp_rules
  return decorator

# Import and organize rule_based dependency functions.
# All rule functions should use @register_kmdp_rule decorator to be properly accessed.

from .manage_rules import topological_sort

# Import local rules first;
from .rules import *
kmdp_rules = topological_sort(_kmdp_precedence)

# Import global rules later
from .global_rules import *
kmdp_rules = topological_sort(_kmdp_precedence)

# kmdp_rules now properly contain rules of orders

from .interface import kmdp_generate, kmdp_recover