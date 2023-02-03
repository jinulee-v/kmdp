dp_analysis_fn = {}
def register_analysis_fn(name):
  def decorator(fn):
    dp_analysis_fn[name] = fn
    return fn
  return decorator

from .accuracy import *
from .confidence import *