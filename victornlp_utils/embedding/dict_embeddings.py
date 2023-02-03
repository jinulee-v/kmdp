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
import json

from . import register_embedding

class EmbeddingDict(nn.Module):
  """
  Abstract Embedding model that creates embedding based on lexicon data.
  """
  
  def __init__(self, config):
    """
    Constructor for EmbeddingDict.
    All embedding files should match the following format:
     [
        {
          "text": "language",
          "pos_tag": "NNG", (optional, any tags)
          "embedding": "1.0 2.1 -0.9 ...", (mandatory if config['from_pretrained']')
          "embed_size": 100, (mandatory if not config['from_pretrained'])
          ...
        },
        ...
     ]
     UNK tokens are mapped to the last of the sequence(i.e. -1 th index).
    
    @param self The object pointer.
    @param config Dictionary. Configuration for the embedding.
    """
    super(EmbeddingDict, self).__init__()
    
    self.file_dir = config['file_directory']
    with open(self.file_dir) as embedding_file:
      self.full_information = json.load(embedding_file)
     
    if 'key' in config:
      self.key = config['key']
    else:
      self.key = 'text'
    self.special_tokens = config['special_tokens']
    assert 'pad' in self.special_tokens
    
    self.itos = []
    self.stoi = {}
    self.embeddings = []
    for i, info in enumerate(self.full_information):
      self.itos.append(info[self.key])
      self.stoi[info[self.key]] = i
      if config['from_pretrained']:
        self.embeddings.append(torch.tensor([float(x) for x in info['embedding'].split()]).unsqueeze(0))
    
    if config['from_pretrained']:
      # Read pretrained embedding
      self.embed_size = self.embeddings[0].size(1)
      for j in self.special_tokens:
        self.embeddings.append(torch.zeros(1, self.embed_size))
      self.embeddings = nn.Embedding.from_pretrained(torch.cat(self.embeddings, 0))
    else:
      # From-Scratch embedding
      self.embed_size = config['embed_size']
      self.embeddings = nn.Embedding(len(self.itos) + len(self.special_tokens), config['embed_size'])
    
    self.requires_grad = bool(config['train'])
    for type, token in self.special_tokens.items():
      assert type in ['unk', 'eos', 'bos', 'pad']
      self.stoi[token] = len(self.itos)
      self.itos.append(token)
  
  def forward(self, inputs):
    """
    @brief creates embedding for every space-separated tokens

    @param inputs VictorNLP format corpus data
    @return embedded Tensor(batch_size, max_length, embed_size)
    """
    device = next(self.parameters()).device

    embedded = []
    for input in inputs:
      has_bos = 1 if 'bos' in self.special_tokens else 0
      has_eos = 1 if 'eos' in self.special_tokens else 0
      word_embedding = torch.zeros(input['word_count'] + has_bos + has_eos, dtype=torch.long).to(device)
      
      if has_bos:
        word_embedding[0] = self.stoi[self.special_tokens['bos']]
      if has_eos:
        word_embedding[-1] = self.stoi[self.special_tokens['eos']]
       
      tokens = [pos[self.key] for pos in input['pos']]
      for word_i, token in enumerate(tokens, has_bos):
        word_embedding[word_i] = self.stoi[self.target(token)]
      
      embedded.append(word_embedding)

    embedded = torch.nn.utils.rnn.pad_sequence(embedded, batch_first=True, padding_value = self.stoi[self.special_tokens['pad']])
    return self.embeddings(embedded)
  
  def target(self, token):
    return token if token in self.itos else self.special_tokens['unk']

@register_embedding('pos-eng')
class EmbeddingPoS_eng(EmbeddingDict):
  """
  PoS tag embedding for English.
  """
@register_embedding('glove-eng')
class EmbeddingGloVe_eng(EmbeddingDict):
  """
  GloVe embedding for English.
  """