"""
@module dict_embedding
Various lexicon-based embedding modules.

Class name format: 
  Embedding(description)_language
Requirements:
  __init__(self, config)
    @param config Dictionary. Configuration for Embedding*_* (free-format)
  self.embed_size
  forward(self, inputs)
    @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
    @return output Tensor(batch, length+1, self.embed_size). length+1 refers to the virtual root.
"""

import torch
import torch.nn as nn

from . import register_embedding
from .dict_embeddings import EmbeddingDict

class EmbeddingDictWordPhr_kor(EmbeddingDict):
  """
  Abstract template for Korean lexicon-style embeddings. Implements Word-phrase level PoS tag selection.
  
  Follows Sejong morpheme tagging format.
  """
  def __init__(self, config):
    super(EmbeddingDictWordPhr_kor, self).__init__(config)
    # Convert to tensor for simple indexing
    self.embeddings = self.embeddings.weight
    self.embed_size *= 2
  
  def forward(self, inputs):
    lexical_morphemes_tag = [
      'NNG', 'NNP', 'NNB', 'NR', 'NP',
      'VV', 'VA', 'VX', 'VCP', 'VCN',
      'MM' 'MAG', 'MAJ',
      'IC', 'SL', 'SN', 'SH', 'XR'
    ]
    embedded = []
    for input in inputs:
      assert 'pos' in input
      
      word_i = has_bos = 1 if 'bos' in self.special_tokens else 0
      has_eos = 1 if 'eos' in self.special_tokens else 0
      word_embedding = torch.zeros(input['word_count'] + has_bos + has_eos, self.embed_size).to(self.embeddings.device)
      
      if has_bos:
        word_embedding[0] = self.embeddings[self.stoi[self.special_tokens['bos']]].repeat(2)
      if has_eos:
        word_embedding[-1] = self.embeddings[self.stoi[self.special_tokens['eos']]].repeat(2)
        
      for word in input['pos']:
        
        for morph in word:
          pos_tag = morph['pos_tag']
          if pos_tag in lexical_morphemes_tag:
            word_embedding[word_i, :self.embed_size//2] = self.embeddings[self.stoi[self.target(morph)]]
          else:
            word_embedding[word_i, self.embed_size//2:] = self.embeddings[self.stoi[self.target(morph)]]
        word_i += 1
            
      embedded.append(word_embedding)
    
    embedded = torch.nn.utils.rnn.pad_sequence(embedded, batch_first=True)
      
    return embedded

@register_embedding('glove-wp-kor')
class EmbeddingGloVeWordPhr_kor(EmbeddingDictWordPhr_kor):
  """
  GloVe embedding for Korean, for word-phrase level tokenization(space).
  """
  def __init__(self, config):
    super(EmbeddingGloVeWordPhr_kor, self).__init__(config)
  
  
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

@register_embedding('pos-wp-kor')
class EmbeddingPoSWordPhr_kor(EmbeddingDictWordPhr_kor):
  """
  PoS tag embedding for Korean, for word-phrase level tokenization(space).
  """
  def __init__(self, config):
    super(EmbeddingPoSWordPhr_kor, self).__init__(config)
  
  def target(self, word):
    if word['pos_tag'] in self.stoi:
      return word['pos_tag']
    else:
      return self.special_tokens['unk']