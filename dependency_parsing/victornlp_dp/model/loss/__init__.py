"""
@module loss
Various loss functions for dependency parsing.

loss_*(parser, inputs)
  @param parser *Parser object. Refer to 'model.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  
  @return Loss value.
"""

dp_loss_fn = {}
def register_loss_fn(name):
  def decorator(fn):
    dp_loss_fn[name] = fn
    return fn
  return decorator

from .loss_local import loss_NLL, loss_XBCE, loss_LH