"""
@module loss_local
Various locally-scoped loss functions for dependency parsing.

Lee(2020) introduces each loss functions briefly. In the same paper, loss_LH was proved to be most effective for Korean data.
"""

import torch
import torch.nn as nn

from . import register_loss_fn

@register_loss_fn('local-nll')
def loss_NLL(parser, inputs, **kwargs):
  """
  Negative Log-Likelihood loss function.
  
  This function only backpropagates from p(head_golden), completely ignoring the possibilities assigned to errors.
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param **kwargs Passed to parser.run().
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs, **kwargs)

  if 'lengths' not in kwargs:
    lengths = torch.zeros(batch_size, dtype=torch.long).detach().to(device)
    for i, input in enumerate(inputs):
      lengths[i] = input['word_count'] + 1
  else:
    lengths = kwargs['lengths']
  max_length = torch.max(lengths)

  golden_heads = torch.zeros((batch_size, max_length, 1), dtype=torch.long).detach().to(device)
  golden_labels = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.long).detach().to(device)
  for i, input in enumerate(inputs):
    for j, arc in enumerate(input['dependency']):
      dep = arc['dep']
      golden_heads[i, dep, 0] = arc['head']
      golden_labels[i, 0, dep, :] = parser.labels_stoi[arc['label']]
  
  loss_arc = arc_attention.squeeze(1).gather(2, golden_heads).squeeze(2)
  loss_type = type_attention.gather(1, golden_labels).squeeze(1).gather(2, golden_heads).squeeze(2)
  
  return -(torch.sum(loss_arc) + torch.sum(loss_type)) / (torch.sum(lengths) - batch_size)

@register_loss_fn('local-xbce')
def loss_XBCE(parser, inputs, **kwargs):
  """
  eXpanded Binary Cross Entropy loss function.
  
  @param parser *Parser object. Refer to 'model.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param **kwargs Passed to parser.run().
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs, **kwargs)
  
  if 'lengths' not in kwargs:
    lengths = torch.zeros(batch_size, dtype=torch.long).detach().to(device)
    for i, input in enumerate(inputs):
      lengths[i] = input['word_count'] + 1
  else:
    lengths = kwargs['lengths']
  max_length = torch.max(lengths)

  if 'mask' not in kwargs:
    golden_heads = torch.zeros((batch_size, max_length, 1), dtype=torch.long).detach().to(device)
    golden_labels = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.long).detach().to(device)
    golden_head_mask = torch.zeros_like(arc_attention, dtype=torch.float).detach().to(device)
    golden_label_mask = torch.zeros_like(type_attention, dtype=torch.float).detach().to(device)
    for i, input in enumerate(inputs):
      for j, arc in enumerate(input['dependency']):
        dep = arc['dep']
        golden_heads[i, dep, 0] = arc['head']
        golden_labels[i, 0, dep, :] = parser.labels_stoi[arc['label']]
        golden_head_mask[i, 0, dep, :lengths[i]] = 1
        golden_head_mask[i, 0, dep, arc['head']] = 0
        golden_label_mask[i, :, dep, arc['head']] = 1
        golden_label_mask[i, parser.labels_stoi[arc['label']], dep, arc['head']] = 0
  else:
    mask = kwargs['mask']
    golden_heads = torch.zeros((batch_size, max_length, 1), dtype=torch.long).detach().to(device)
    golden_labels = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.long).detach().to(device)
    golden_head_mask = kwargs['mask'].clone().detach().to(device)
    golden_label_mask = torch.zeros_like(type_attention, dtype=torch.float).detach().to(device)
    for i, input in enumerate(inputs):
      for j, arc in enumerate(input['dependency']):
        dep = arc['dep']
        golden_heads[i, dep, 0] = arc['head']
        golden_labels[i, 0, dep, :] = parser.labels_stoi[arc['label']]
        golden_head_mask[i, 0, dep, arc['head']] = 0
        golden_label_mask[i, :, dep, arc['head']] = mask[i, 0, dep, arc['head']]
        golden_label_mask[i, parser.labels_stoi[arc['label']], dep, arc['head']] = 0  

  loss_arc = torch.sum((arc_attention).squeeze(1).gather(2, golden_heads).squeeze(2))
  loss_arc += torch.sum(torch.log(1+(1e-6)-torch.exp(arc_attention)) * golden_head_mask)
  
  loss_type = torch.sum((type_attention).gather(1, golden_labels).squeeze(1).gather(2, golden_heads).squeeze(2))
  loss_type += torch.sum(torch.log(1+(1e-6)-torch.exp(type_attention)) * golden_label_mask)
  
  return -(loss_arc + loss_type) / (torch.sum(lengths) - batch_size)
  
@register_loss_fn('local-lh')
def loss_LH(parser, inputs, **kwargs):
  """
  Local Hinge loss function.
  
  @param parser *Parser object. Refer to 'model.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param **kwargs Passed to parser.run().
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs, **kwargs)
  
  if 'lengths' not in kwargs:
    lengths = torch.zeros(batch_size, dtype=torch.long).detach().to(device)
    for i, input in enumerate(inputs):
      lengths[i] = input['word_count'] + 1
  else:
    lengths = kwargs['lengths']
  max_length = torch.max(lengths)

  
  if 'mask' not in kwargs:
    golden_heads = torch.zeros((batch_size, max_length, 1), dtype=torch.long).detach().to(device)
    golden_labels = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.long).detach().to(device)
    golden_head_mask = torch.zeros_like(arc_attention, dtype=torch.float).detach().to(device)
    golden_label_mask = torch.zeros_like(type_attention, dtype=torch.float).detach().to(device)
    for i, input in enumerate(inputs):
      for j, arc in enumerate(input['dependency']):
        dep = arc['dep']
        golden_heads[i, dep, 0] = arc['head']
        golden_labels[i, 0, dep, :] = parser.labels_stoi[arc['label']]
        golden_head_mask[i, 0, dep, :lengths[i]] = 1
        golden_head_mask[i, 0, dep, arc['head']] = 0
        golden_label_mask[i, :, dep, arc['head']] = 1
        golden_label_mask[i, parser.labels_stoi[arc['label']], dep, arc['head']] = 0
  else:
    mask = kwargs['mask']
    golden_heads = torch.zeros((batch_size, max_length, 1), dtype=torch.long).detach().to(device)
    golden_labels = torch.zeros((batch_size, 1, max_length, max_length), dtype=torch.long).detach().to(device)
    golden_head_mask = kwargs['mask'].clone().detach().to(device)
    golden_label_mask = torch.zeros_like(type_attention, dtype=torch.float).detach().to(device)
    for i, input in enumerate(inputs):
      for j, arc in enumerate(input['dependency']):
        dep = arc['dep']
        golden_heads[i, dep, 0] = arc['head']
        golden_labels[i, 0, dep, :] = parser.labels_stoi[arc['label']]
        golden_head_mask[i, 0, dep, arc['head']] = 0
        golden_label_mask[i, :, dep, arc['head']] = mask[i, 0, dep, arc['head']]
        golden_label_mask[i, parser.labels_stoi[arc['label']], dep, arc['head']] = 0  
  
  golden_head_scores = arc_attention.squeeze(1).gather(2, golden_heads).squeeze(2)
  hinge_head_scores = torch.max((arc_attention + golden_head_mask * 100000).squeeze(1), dim=2)[0] - 100000
  loss_arc = torch.sum(torch.maximum(hinge_head_scores - golden_head_scores + torch.ones_like(golden_head_scores), torch.zeros_like(golden_head_scores)))
  
  golden_label_scores = type_attention.gather(1, golden_labels).squeeze(1).gather(2, golden_heads).squeeze(2)
  hinge_label_scores = torch.max(type_attention + golden_label_mask * 100000, dim=1)[0].gather(2, golden_heads).squeeze(2) - 100000
  loss_type = torch.sum(torch.maximum(hinge_label_scores - golden_label_scores + torch.ones_like(golden_label_scores), torch.zeros_like(golden_label_scores)))
  
  return (loss_arc + loss_type) / (torch.sum(lengths) - batch_size)