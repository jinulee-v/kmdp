"""
@module loss
Various loss functions for dependency parsing.

loss_*(model, inputs)
  @param model Model object. Refer to '..model' for more details.
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  
  @return Loss value.
"""

re_loss_fn = {}
def register_loss_fn(name):
  def decorator(fn):
    re_loss_fn[name] = fn
    return fn
  return decorator

from .loss_mtre_sentence import *
from .loss_nll import *