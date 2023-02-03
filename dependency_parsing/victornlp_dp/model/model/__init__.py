dp_model = {}
def register_model(name):
  def decorator(cls):
    dp_model[name] = cls
    return cls
  return decorator

from .LeftToRightParser import *
from .DeepBiaffineParser import *