"""
@module bert_embedding
Various transformer-based embedding modules.

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
import itertools

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from kobert_transformers import get_tokenizer

from . import register_embedding

@register_embedding('kobert')
class EmbeddingBERTWordPhr_kor(nn.Module):
  """
  Word-phrase level Embedding model using KoBERT(SKT-brain, 2019).
  Concatenates the hidden state of initial & final wordpiece tokens.
  """
  def __init__(self, config):
    """
    Constructor for EmbeddingBERTWordPhr_kor.
    
    @param self The object pointer.
    @param config Dictionary. Configuration for EmbeddingBERTWordPhr_kor
    """
    super(EmbeddingBERTWordPhr_kor, self).__init__()
    self.tokenizer = get_tokenizer()
    self.model = BertModel.from_pretrained('monologg/kobert')
    self.embed_size = 1536
    self.special_tokens = config['special_tokens']
    self.fine_tune = bool(config['train'])
    if self.fine_tune:
      self.model.train()
      self.model.requires_grad = True
    else:
      self.model.eval()
      self.model.requires_grad = False

  def forward(self, inputs):
    """
    Overridden forward().
    
    @param self The object pointer.
    @param inputs List of dictionary. Refer to 'corpus.py' for more details.
    
    @return Tensor(batch, length, self.embed_size)
    """
    device = next(self.parameters()).device
    
    tokens_list = []
    sentences = []
    attention_mask = []
    for input in inputs:
      tokens = self.tokenizer.tokenize('[CLS] '+input['text'].replace(' _ ', ' ').replace('_', '')+' [SEP]')
      tokens_list.append(tokens)
      tokens = self.tokenizer.convert_tokens_to_ids(tokens)
      sentences.append(torch.tensor(tokens, dtype=torch.long))
      attention_mask.append(torch.ones([len(tokens)], dtype=torch.long))
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device)

    if self.fine_tune:
      output = self.model(sentences, attention_mask, torch.zeros_like(attention_mask))['last_hidden_state']
    else:
      with torch.no_grad():
        output = self.model(sentences, attention_mask, torch.zeros_like(attention_mask))['last_hidden_state']
    embedded = []
    temp = None
    for i, tokens in enumerate(tokens_list):
      embedded.append([])
      for j in range(len(tokens)):
        if tokens[j] == '[SEP]':
          embedded[i].append(output[i][j].repeat(2).unsqueeze(0))
          break
        if tokens[j] == '[CLS]' or tokens[j].startswith('▁'):
          temp = output[i][j]
        if j+1 == len(tokens) or tokens[j+1] == '[SEP]' or tokens[j+1].startswith('▁'):
          temp = torch.cat([temp, output[i][j]], 0)
          embedded[i].append(temp.unsqueeze(0))
          temp = None
      embedded[i] = torch.cat(embedded[i], 0)
      if 'bos' not in self.special_tokens:
        embedded[i] = embedded[i][1:]
      if 'eos' not in self.special_tokens:
        embedded[i] = embedded[i][:-1]
    embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
    return embedded
