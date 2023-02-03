"""
@module attention

Implementation of various attention modules.
- Bilinear Attention
"""

import torch
import torch.nn as nn

class BilinearAttention(nn.Module):
  """
  Bilinear attention module.
  """

  def __init__(self, encoder_size, decoder_size, label_size):
    """
    Constructor for BilinearAttention.
    
    @param self The object pointer.
    @param encoder_size Integer. Hidden size of the LSTM encoder.
    @param decoder_size Integer. Hidden size of the LSTM decoder.
    @param label_size Integer. Size of label lists.
    """
    super(BilinearAttention, self).__init__()
    self.W = nn.Parameter(torch.zeros(label_size, decoder_size, encoder_size))
    self.u = nn.Parameter(torch.zeros(label_size, encoder_size))
    self.v = nn.Parameter(torch.zeros(label_size, decoder_size))
    self.b = nn.Parameter(torch.zeros(label_size))
    
    nn.init.xavier_uniform_(self.W)
    nn.init.xavier_uniform_(self.u)
    nn.init.xavier_uniform_(self.v)
    

  def forward(self, encoder_h, decoder_h, lengths):
    """
    Overridden forward().
    
    @param encoder_h Tensor(batch, length, encoder_size). Last layer of encoder hidden states.
    @param decoder_h Tensor(batch, length, decoder_size). Last layer of decoder hidden states.
    @param lengths LongTensor(batch). Contains each sequence length per inputs in batch.
    
    @return Tensor(batch, label_size, length, length). 
    """
    
    # Attention
    output = torch.matmul(torch.matmul(decoder_h.unsqueeze(1), self.W), encoder_h.unsqueeze(1).transpose(2, 3))
    output = output + torch.matmul(self.u, encoder_h.transpose(1,2)).unsqueeze(2)
    output = output + torch.matmul(self.v, decoder_h.transpose(1,2)).unsqueeze(2)
    output = output + self.b.unsqueeze(1).unsqueeze(2)
    
    return output
