"""
@module run_mtre

Contains:
- Greedy decoding(Argmax) for dependency parsing
- Max-K candidates for entity type labels
- Single guess for relation labels
"""

import torch

from . import register_run_fn

@register_run_fn('argmax')
def run_argmax(model, inputs, config, **kwargs):
  device = next(model.parameters()).device
  batch_size = len(inputs)

  relations, relation_scores= model.run(inputs, **kwargs)
  
  # Relation labeling
  _, relation_max = torch.max(relation_scores, dim=1)
  for relation, relation_id in zip(relations, relation_max):
    if 'label' in relation:
      key = 'label_predict'
    else:
      key = 'label'
    relation[key] = model.re_labels[relation_id]

  return inputs
