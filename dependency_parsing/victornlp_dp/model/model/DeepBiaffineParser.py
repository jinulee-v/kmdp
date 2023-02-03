"""
@module DeepBiaffineParser
Implementation of Deep Biaffine Parser(Dozat, 2018)
"""

import torch
import torch.nn as nn

from ...victornlp_utils.module import *

from . import register_model

@register_model('deep-biaff-parser')
class DeepBiaffineParser(nn.Module):
  """
  Deep-Biaffine parser module.
  """
  
  def __init__(self, embeddings, labels, config):
    """
    Constructor for Deep-Biaffine parser.
    
    @param self The object pointer.
    @param embeddings List of Embedding-like objects. Refer to 'embedding.py'' for more details.
    @param labels List of dependency type labels(strings)
    @param config Dictionary config file accessed with 'DeepBiaffineParser' key.
    """
    super(DeepBiaffineParser, self).__init__()

    self.config = config
    self.hidden_size = config['hidden_size']

    # Embedding layer
    self.embeddings = nn.ModuleList(embeddings)
    input_size = 0
    for embedding in embeddings:
      input_size += embedding.embed_size

    # Model layer (Bi-LSTM Encoder)
    self.encoder = nn.LSTM(input_size, self.hidden_size, config['encoder']['num_layers'], batch_first=True, dropout=config['encoder']['dropout'], bidirectional=True)
    
    self.arc_encoder = nn.Sequential(
      nn.Linear(self.hidden_size*2, config['arc_size']),
      nn.ELU(),
      nn.Dropout(config['encoder']['dropout'])
    )
    self.type_encoder = nn.Sequential(
      nn.Linear(self.hidden_size*2, config['type_size']),
      nn.ELU(),
      nn.Dropout(config['encoder']['dropout'])
    )
    
    self.arc_att = BilinearAttention(config['arc_size'], config['arc_size'], 1)
    self.type_att = BilinearAttention(config['type_size'], config['type_size'],  len(labels))

    # Prediction layer
    self.labels = labels  # type labels
    self.labels_stoi = {}
    for i, label in enumerate(self.labels):
      self.labels_stoi[label] = i
      
  def run(self, inputs, lengths=None, mask=None):
    """
    Runs the parser and obtain scores for each possible arc and type.
    
    @param self The object pointer.
    @param inputs List of dictionaries. Refer to 'corpus.py'' for more details.
    @param lengths Tensor(batch_size). Contains length information, but for default it uses word_count data.
    @param mask Tensor(batch_size, 1, max_length, max_length). Contains mask information, but for default it uses word_count data
    @return arc_attention Tensor(batch, 1, length, length). arc_attention[i][0][j][k] contains the log(probability) of arc j->k among j->* arcs in batch i.
    @return type_attention Tensor(batch, type, length, length)
    """
    batch_size = len(inputs)
    if lengths is None:
      lengths = torch.zeros(batch_size, dtype=torch.long).detach()
      for i, input in enumerate(inputs):
        lengths[i] = input['word_count'] + 1 # DP requires [ROOT] node; index 0 is given
    else:
      assert lengths.dim()==1 and lengths.size(0) == batch_size
      lengths = lengths.detach()
    
    # Embedding
    embedded = []
    for embedding in self.embeddings:
      embedded.append(embedding(inputs))
    embedded = torch.cat(embedded, dim=2)
    
    # Encoder
    encoder_h, _ = self.encoder(embedded)

    # Biaffine Attention
    encoder_h_arc = self.arc_encoder(encoder_h)
    encoder_h_type = self.type_encoder(encoder_h)
    arc_attention = self.arc_att(encoder_h_arc, encoder_h_arc, lengths)
    type_attention = self.type_att(encoder_h_type, encoder_h_type, lengths)
    
    # Masking & log_softmax
    if mask is None:
      mask = torch.zeros_like(arc_attention).detach()
      for i, length in enumerate(lengths):
        for j in range(1, length):
          mask[i, 0, j, :length] = 1
    else:
      assert mask.size()==arc_attention.size()
      mask = mask.detach()
  
    arc_attention = arc_attention - ((1-mask) * 100000)
    arc_attention = nn.functional.log_softmax(arc_attention, 3)
  
    type_attention = type_attention - ((1-mask) * 100000)
    type_attention = nn.functional.log_softmax(type_attention, 1)
    
    return arc_attention, type_attention
    