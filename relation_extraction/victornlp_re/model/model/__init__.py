"""
@module model
Various re analysis models.

class *RelExtract(nn.Module):
  run(model, inputs, **kwargs)
    @param model Model object. Refer to '.*' for more details.
    @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
    @param **kwargs Other keywords required for each model.
"""

re_model = {}
def register_model(name):
  def decorator(cls):
    re_model[name] = cls
    return cls
  return decorator

from .MultiTaskRelExtract import *
from .PrunedDependencyRelExtract import *