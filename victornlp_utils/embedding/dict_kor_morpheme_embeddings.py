"""
@module dict_kor_morpheme_embeddings.py
"""

import torch
import torch.nn as nn

from . import register_embedding
from .dict_embeddings import EmbeddingDict
class EmbeddingDictMorph_kor(EmbeddingDict):
  """
  Abstract template for Korean lexicon-style embeddings. Implements Morpheme level PoS tag selection.
  
  Follows Sejong morpheme tagging format.
  """
  def __init__(self, config):
    super(EmbeddingDictMorph_kor, self).__init__(config)
    # Convert to tensor for simple indexing
  
  def forward(self, inputs):
    embedded = []
    for input in inputs:
      assert 'pos' in input
      
      has_bos = 1 if 'bos' in self.special_tokens else 0
      has_eos = 1 if 'eos' in self.special_tokens else 0
      token_idx = torch.zeros(input['pos'][-1][-1]['id'] + has_bos + has_eos, dtype=torch.long, device=self.embeddings.weight.device)
      
      if has_bos:
        token_idx[0] = self.stoi[self.special_tokens['bos']]
      if has_eos:
        token_idx[-1] = self.stoi[self.special_tokens['eos']]
      
      for word in input['pos']:
        for morph in word:
          token_idx[morph['id'] + has_bos - 1] = self.stoi[self.target(morph)]
            
      embedded.append(self.embeddings(token_idx))
    
    embedded = torch.nn.utils.rnn.pad_sequence(embedded, batch_first=True)
      
    return embedded

@register_embedding('glove-morph-kor')
class EmbeddingGloVeMorph_kor(EmbeddingDictMorph_kor):
  """
  GloVe embedding for Korean, for morpheme level tokenization.
  """
  def __init__(self, config):
    super(EmbeddingGloVeMorph_kor, self).__init__(config)
  
  def target(self, word):
    replace = {
      'MMD': 'MM',
      'MMN': 'MM',
      'MMA': 'MM'
    }
    if word['pos_tag'] in replace:
      word['pos_tag'] = replace[word['pos_tag']]
    joined = word['text'] + '/' + word['pos_tag']
    if joined in self.stoi:
      return joined
    else:
      return self.special_tokens['unk']


@register_embedding('pos-morph-kor')
class EmbeddingPoSMorph_kor(EmbeddingDictMorph_kor):
  """
  PoS tag embedding for Korean, for morpheme level tokenization.
  """
  def __init__(self, config):
    super(EmbeddingPoSMorph_kor, self).__init__(config)
  
  def target(self, word):
    if word['pos_tag'] in self.stoi:
      return word['pos_tag']
    else:
      return self.special_tokens['unk']