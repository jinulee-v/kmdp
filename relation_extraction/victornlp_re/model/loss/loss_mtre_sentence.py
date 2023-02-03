"""
@module loss_mtre_sentence

Implements loss functions(pretraining for transfer learning and the actual RE training phase).
"""

import torch
import torch.nn.functional as F

from . import register_loss_fn


def loss_MTRE_DP_NLL(arc_attention, inputs, **kwargs):
  """
  Negative Log-Likelihood loss function for the dependency parsing.
  
  This function only backpropagates from p(head_golden), completely ignoring the possibilities assigned to errors.
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param **kwargs Passed to parser.run().
  """
  device = arc_attention.device
  batch_size = len(inputs)

  if 'lengths' not in kwargs:
    lengths = torch.zeros(batch_size, dtype=torch.long).detach().to(device)
    for i, input in enumerate(inputs):
      # lengths[i] = input['word_count'] + 1
      lengths[i] = 1 + sum([len(wp) for wp in input['pos']])
  else:
    lengths = kwargs['lengths']
  max_length = torch.max(lengths)

  golden_heads = torch.zeros((batch_size, max_length, 1), dtype=torch.long).detach().to(device)
  for i, input in enumerate(inputs):
    tree = input['dependency']
    new_tree = [None] * max_length
    for arc in tree:
      new_tree[arc['dep']] = arc
    tree = new_tree
    for j, arc in enumerate(tree):
      if arc is None:
        continue
      dep = arc['dep']
      golden_heads[i, dep, 0] = arc['head']
  
  loss_arc = arc_attention.squeeze(1).gather(2, golden_heads).squeeze(2)
  
  return -torch.sum(loss_arc) / (torch.sum(lengths) - batch_size)


@register_loss_fn('mtre-sentence-pretrain')
def loss_MTRE_sentence_pretrain(model, inputs, **kwargs):
  """
  Loss for pretraining MTRE model with only auxiliary tasks.
  """
  device = next(model.parameters()).device
  batch_size = len(inputs)
  
  kwargs['pretrain'] = True
  relations, relation_scores, arc_attention, entities, entity_type_scores = model.run(inputs, **kwargs)
  assert relations is None and relation_scores is None

  # Apply NLL loss for Dependency Parsing.
  dp_loss = loss_MTRE_DP_NLL(arc_attention, inputs)
  
  # Apply Normalized-Hinge loss for Entity Type Labeling.
  entity_type_loss = torch.zeros(1, device=device)
  count = 0
  for e_i, entity in enumerate(entities):
    for label in entity['label']:
      if label in model.etl_labels:
        score = entity_type_scores[e_i][model.etl_labels_stoi[label]]
        entity_type_loss += torch.max(torch.zeros(1, device=device), 3-score)
        count += 1
  entity_type_loss /= count

  return dp_loss + entity_type_loss * 2


@register_loss_fn('mtre-sentence')
def loss_MTRE_sentence(model, inputs, **kwargs):
  """
  Loss for full-training MTRE model with RE and auxiliary tasks together.
  """
  device = next(model.parameters()).device
  batch_size = len(inputs)
  
  kwargs['pretrain'] = False
  relations, relation_scores, arc_attention, entities, entity_type_scores = model.run(inputs, **kwargs)
  assert relations is not None and relation_scores is not None

  # Apply NLL loss for Dependency Parsing.
  dp_loss = loss_MTRE_DP_NLL(arc_attention, inputs)
  
  # Apply Normalized-Hinge loss for Entity Type Labeling.
  entity_type_loss = torch.zeros(1, device=device)
  count = 0
  for e_i, entity in enumerate(entities):
    for label in entity['label']:
      if label in model.etl_labels:
        score = entity_type_scores[e_i][model.etl_labels_stoi[label]]
        entity_type_loss += torch.max(torch.zeros(1, device=device), 3-score)
        count += 1
  entity_type_loss /= count
  
  # Apply NLL loss for Relation Extraction.
  # Generate label
  re_golden = torch.zeros(len(relations), dtype=torch.long, device=device)
  for r_i, relation in enumerate(relations):
    re_golden[r_i] = model.re_labels_stoi[relation['label']]
  re_loss = F.nll_loss(relation_scores, re_golden, reduction='mean')

  return re_loss + (dp_loss + entity_type_loss) * 0.5