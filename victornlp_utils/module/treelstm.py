"""
@module treelstm

Implementation of Child-sum Tree-LSTM and its variants.
- Original
    Kai, T. S. et al.(2015) Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
- Edge-labeled version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChildSumTreeLSTM(nn.Module):
  """
  @class ChildSumTreeLSTM

  Implementation of Child-Sum Tree LSTM with VictorNLP dependency tree input.
  Dependency tree format:
  [
    {
      'dep': Integer,
      'head': Integer,
      'label': String
    },
    ...
  ]
  """

  def __init__(self, input_size, hidden_size):
    """
    Constructor for ChildSumTreeLSTM.
    
    @param self The object pointer.
    @param input_size Integer. Input size(word embedding).
    @param hidden_size Integer. Hidden size of theTreeLSTM unit.
    """
    super(ChildSumTreeLSTM, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size

    self.W_i = nn.Linear(self.input_size, self.hidden_size)
    self.W_o = nn.Linear(self.input_size, self.hidden_size)
    self.W_u = nn.Linear(self.input_size, self.hidden_size)
    self.W_f = nn.Linear(self.input_size, self.hidden_size)
    self.U_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.U_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.U_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.U_f = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
  
  def _forward_node(self, j, children, x, h, c):
    """
    Forward per node.
    
    @param self The object pointer.
    @param j Integer. Index of current node.
    @param children LongTensor. Indices of the children of the current node.
    @param x Tensor(length, input_size). Input embedding of the current sentence.
    @param h List[Tensor(hidden_size)]. Hidden state of the current sentence.
    @param c List[Tensor(hidden_size)]. Cell state of the current sentence.
    """

    if len(children) > 0:
      h_j_sum = torch.sum(torch.cat([h[child].unsqueeze(0) for child in children], dim=0), dim=0)
    else:
      h_j_sum = torch.zeros_like(h[j])

    i_j = torch.sigmoid(self.W_i(x[j]) + self.U_i(h_j_sum))
    o_j = torch.sigmoid(self.W_o(x[j]) + self.U_o(h_j_sum))
    u_j = torch.tanh(self.W_u(x[j]) + self.U_u(h_j_sum))
    assert i_j.size() == o_j.size() == u_j.size()

    c[j] = i_j * u_j
    for child in children:
      c[j] += torch.sigmoid((self.W_f(x[j, :]) + self.U_f(h[child]))) * c[child]
    h[j] = o_j * torch.tanh(c[j])
  
  def _forward_postorder(self, j, children_list, x, h, c, device):
    """
    Postorder traversal of the given tree.
    """
    children = children_list[j]
    if len(children) != 0:
      for child in children:
        self._forward_postorder(child, children_list, x, h, c, device)
    
    self._forward_node(j, children, x, h, c)
  
  def forward(self, dp_tree, input_embeddings, root):
    """
    forward() override for nn.Module.

    @param dp_tree VictorNLP depednency tree format
    @param input_embeddings Tensor(length, input_size).
    @param root Integer. Starting point for _forward_postorder
    @return h Tensor(length, hidden_size). Hidden state of the Tree-LSTM.
    @return c Tensor(length, hidden_size). Cell state of the Tree-LSTM.
    """
    device = input_embeddings.device

    # Create children_list
    children_list = [[] for _ in range(input_embeddings.size(0))]
    for arc in dp_tree:
      children_list[arc['head']].append(arc['dep'])
    
    # Create empty hidden/cell state matrices
    h = [torch.zeros(self.hidden_size, device=device) for _ in range(input_embeddings.size(0))]
    c = [torch.zeros(self.hidden_size, device=device) for _ in range(input_embeddings.size(0))]

    # Recursive call
    self._forward_postorder(root, children_list, input_embeddings, h, c, device)

    h = torch.cat([h_x.unsqueeze(0) for h_x in h], dim=0)
    c = torch.cat([c_x.unsqueeze(0) for c_x in c], dim=0)

    return h, c