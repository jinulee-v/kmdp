"""
@module TreeLSTMSTS

Implements STS using Child-sum Tree LSTM with dependency parsing results.
> Kai, T. S. et al.(2015) Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
"""

import torch
import torch.nn as nn

from . import register_model

from ...victornlp_utils.module import ChildSumTreeLSTM

@register_model('tree-lstm')
class TreeLSTMSTS(nn.Module):
  """
  @class TreeLSTMSTS

  STS analysis with TreeLSTM
  """

  def __init__(self, embeddings, labels, config):
    """
    Constructor for Deep-Biaffine parser.
    
    @param self The object pointer.
    @param embeddings List of Embedding-like objects. Refer to 'embedding.py'' for more details.
    @param labels List of sts labels(strings)
    @param config Dictionary config file accessed with 'TreeLSTMSTS' key.
    """
    super(TreeLSTMSTS, self).__init__()

    # Embedding layer
    self.embeddings = nn.ModuleList(embeddings)
    input_size = 0
    for embedding in embeddings:
      input_size += embedding.embed_size
    self.input_size = input_size
    self.hidden_size = config['hidden_size']
    self.classifier_size = config['classifier_size']
    self.r_size = config['r_size']

    # Model layer (Tree-LSTM Encoder)
    self.encoder = ChildSumTreeLSTM(self.input_size, self.hidden_size)
    
    # Prediction layer
    self.W_mul = nn.Linear(self.hidden_size, self.classifier_size)
    self.W_diff = nn.Linear(self.hidden_size, self.classifier_size)
    self.prediction = nn.Sequential(
      nn.Linear(self.classifier_size * 2, self.r_size),
      nn.LogSoftmax(dim=1)
    )
  
  def run(self, inputs_a, inputs_b):
    """
    Runs the model and obtain softmax-ed distribution for each possible labels.
    
    @param self The object pointer.
    @param inputs_a List of dictionaries. Refer to 'dataset.py' for more details.
    @param inputs_b List of dictionaries(. Refer to 'dataset.py' for more details.
    @return scores Tensor(batch, labels). scores[i][j] contains the log(probability) of i-th sentence in batch having j-th label.
    """
    batch_size = len(inputs_a)
    
    # Embedding
    embedded_1 = []
    embedded_2 = []
    for embedding in self.embeddings:
      embedded_1.append(embedding(inputs_a))
      embedded_2.append(embedding(inputs_b))
    embedded_1 = torch.cat(embedded_1, dim=2)
    embedded_2 = torch.cat(embedded_2, dim=2)
    
    # Run TreeLSTM and prediction
    results_1 = []
    results_2 = []
    for i, tuple in enumerate(zip(inputs_a, inputs_b)):
      hidden_state_1, _ = self.encoder(tuple[0]['dependency'], embedded_1[i], 0)
      hidden_state_2, _ = self.encoder(tuple[1]['dependency'], embedded_2[i], 0)
      results_1.append(hidden_state_1[0].unsqueeze(0))
      results_2.append(hidden_state_2[0].unsqueeze(0)) # Pick only hidden state of the ROOT.
    results_1 = torch.cat(results_1, dim=0)
    results_2 = torch.cat(results_2, dim=0)

    # results: Tensor(batch_size, hidden_size)
    h_mul = results_1 * results_2
    h_diff = torch.abs(results_1 - results_2)
    h_s = torch.sigmoid(torch.cat((self.W_mul(h_mul), self.W_diff(h_diff)), dim=1))
    p_hat = self.prediction(h_s)
    # p_hat: Tensor(batch_size, r_size)

    return p_hat
