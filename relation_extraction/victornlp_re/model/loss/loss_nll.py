"""
@module loss_mtre_sentence

Implements loss functions(pretraining for transfer learning and the actual RE training phase).
"""

import torch
import torch.nn.functional as F

from . import register_loss_fn


@register_loss_fn('nll')
def loss_nll(model, inputs, **kwargs):
  """
  Loss for full-training MTRE model with RE and auxiliary tasks together.
  """
  device = next(model.parameters()).device
  batch_size = len(inputs)
  
  relations, relation_scores = model.run(inputs, **kwargs)

  # Apply NLL loss for Relation Extraction.
  # Generate label
  re_golden = torch.zeros(len(relations), dtype=torch.long, device=device)
  for r_i, relation in enumerate(relations):
    re_golden[r_i] = model.re_labels_stoi[relation['label']]
  
  re_loss = F.nll_loss(relation_scores, re_golden, reduction='mean')

  return re_loss