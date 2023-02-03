import warnings

embeddings = {}
def register_embedding(name):
  """
  @brief decorator for embedding classes.
  """
  def decorator(cls):
    embeddings[name] = cls
    return cls
  return decorator

try:
  from .bert_embeddings import *
  from .bert_korbert_embedding import *
  try:
    from .bert_kobert_embedding import *
  except:
    warnings.warn('KoBERT embedding cannot be loaded:\n  pip install kobert_transformers sentencepiece')
except:
  warnings.warn('BERT embeddings cannot be loaded:\n  pip install transformers')

from .dict_embeddings import *
from .dict_kor_morpheme_embeddings import *
from .dict_kor_wordphrase_embeddings import *