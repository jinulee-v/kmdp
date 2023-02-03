import warnings

pos_taggers = {}
def register_pos_tagger(name):
  """
  @brief decorator for embedding classes.
  """
  def decorator(fn):
    pos_taggers[name] = fn
    return fn
  return decorator

try:
  from .korean import *
except Exception as e:
  warnings.warn('Khaiii PoS tagger cannot be loaded:\n' + str(e))

try:
  from .english import *
except Exception as e:
  warnings.warn('NLTK PoS tagger cannot be loaded:\n  pip install nltk')