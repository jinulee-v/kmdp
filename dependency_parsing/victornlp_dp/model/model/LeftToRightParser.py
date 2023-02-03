"""
@module LeftToRightParser
Implementation of Left-To-Right Parser(Fernandez-Gonzalez, 2019)
"""

import torch
import torch.nn as nn

from ...victornlp_utils.module import *

from . import register_model

@register_model('left-to-right-parser')
class LeftToRightParser(nn.Module):
  """
  Left-To-Right parser module.
  """

  def __init__(self, embeddings, labels, config):
    """
    Constructor for Left-To-Right parser.
    
    @param self The object pointer.
    @param embeddings List of Embedding-like objects. Refer to 'embedding.py'' for more details.
    @param labels List of dependency type labels(strings)
    @param config Dictionary config file accessed with 'LeftToRightParser' key.
    """
    super(LeftToRightParser, self).__init__()

    self.config = config
    self.hidden_size = config['hidden_size']

    # Embedding layer
    self.embeddings = nn.ModuleList(embeddings)
    input_size = 0
    for embedding in embeddings:
      input_size += embedding.embed_size

    # Model layer (seq2seq, bilinear)
    self.encoder = nn.LSTM(input_size, self.hidden_size, config['encoder']['num_layers'], batch_first=True, dropout=config['encoder']['dropout'], bidirectional=True)
    self.hx_dense = nn.Linear(self.hidden_size*2, self.hidden_size)
    self.decoder = nn.LSTM(self.hidden_size*2, self.hidden_size, config['decoder']['num_layers'], batch_first=True, dropout=config['decoder']['dropout'], bidirectional=False)
    
    self.arc_encoder = nn.Sequential(
      nn.Linear(self.hidden_size*2, config['arc_size']),
      nn.ELU(),
      nn.Dropout(config['encoder']['dropout'])
    )
    self.arc_decoder = nn.Sequential(
      nn.Linear(self.hidden_size, config['arc_size']),
      nn.ELU(),
      nn.Dropout(config['decoder']['dropout'])
    )
    self.type_encoder = nn.Sequential(
      nn.Linear(self.hidden_size*2, config['type_size']),
      nn.ELU(),
      nn.Dropout(config['encoder']['dropout'])
    )
    self.type_decoder = nn.Sequential(
      nn.Linear(self.hidden_size, config['type_size']),
      nn.ELU(),
      nn.Dropout(config['decoder']['dropout'])
    )
    
    self.arc_att = BilinearAttention(config['arc_size'], config['arc_size'], 1)
    self.type_att = BilinearAttention(config['type_size'], config['type_size'],  len(labels))

    # Prediction layer
    self.labels = labels  # type labels
    self.labels_stoi = {}
    for i, label in enumerate(self.labels):
      self.labels_stoi[label] = i

  def _transform_LSTM_hidden_states(self, hn, cn):
    """
    Transforms LSTM state from encoder to decoder.
    Refers from Stack-Pointer Net(Ma, 2019).
    
    @param hn LSTM hidden state.
    @param cn LSTM cell state.
    """
    # take the last layers
    # [2, batch, hidden_size]
    cn = cn[-2:]
    # hn [2, batch, hidden_size]
    _, batch, hidden_size = cn.size()
    # first convert cn t0 [batch, 2, hidden_size]
    cn = cn.transpose(0, 1).contiguous()
    # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
    cn = cn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
    # take hx_dense to [1, batch, hidden_size]
    cn = self.hx_dense(cn)
    # [decoder_layers, batch, hidden_size]
    decoder_layers = self.config['decoder']['num_layers']
    if decoder_layers > 1:
     cn = torch.cat([cn, cn.data.new(decoder_layers - 1, batch, hidden_size).zero_()], dim=0)
    # hn is tanh(cn)
    hn = torch.tanh(cn)
    return hn, cn

  def run(self, inputs, lengths=None, mask=None):
    """
    Runs the parser and obtain scores for each possible arc and type.
    
    @param self The object pointer.
    @param inputs List of dictionaries. Refer to 'corpus.py'' for more details.
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
    
    # Seq2Seq
    encoder_h, (encoder_h_n, encoder_c_n) = self.encoder(embedded)
    decoder_h_0, decoder_c_0 = self._transform_LSTM_hidden_states(encoder_h_n, encoder_c_n)
    decoder_h, _ = self.decoder(encoder_h, (decoder_h_0, decoder_c_0))

    # Bilinear Attention
    encoder_h_arc = self.arc_encoder(encoder_h)
    decoder_h_arc = self.arc_decoder(decoder_h)
    encoder_h_type = self.type_encoder(encoder_h)
    decoder_h_type = self.type_decoder(decoder_h)
    arc_attention = self.arc_att(encoder_h_arc, decoder_h_arc, lengths)
    type_attention = self.type_att(encoder_h_type, decoder_h_type, lengths)
    
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
