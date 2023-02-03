"""
@module model
Various sts analysis models.

class *STS(nn.Module):
  run(model, inputs, **kwargs)
    @param model Model object. Refer to '.*' for more details.
    @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
    @param **kwargs Other keywords required for each model.
"""

sts_model = {}
def register_model(name):
  def decorator(cls):
    sts_model[name] = cls
    return cls
  return decorator

from .TreeLSTMSTS import *