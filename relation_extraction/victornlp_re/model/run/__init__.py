"""
@module run
Various running functions that adds answer data to the input.

run(model, inputs, config)
  @param model Model object. Refer to '.*' for more details.
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  @param config Dictionary config file accessed with 'run' key.
"""

re_run_fn = {}
def register_run_fn(name):
  def decorator(fn):
    re_run_fn[name] = fn
    return fn
  return decorator

from .run_mtre import *
from .run_argmax import *