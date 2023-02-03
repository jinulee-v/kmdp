"""
@module run_mtre

Contains:
- Greedy decoding(Argmax) for dependency parsing
- Max-K candidates for entity type labels
- Single guess for relation labels
"""

import torch

from . import register_run_fn

def parse_greedy(arc_attention, inputs, config, **kwargs):
  """
  Simple argmax parsing. No topologocal restrictions are applied during parsing, thus may generate improper structures. Use for testing.
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  @param **kwargs Passed to parser.run().
  
  @return 'inputs' dictionary with parse tree information added.
  """
  device = arc_attention.device
  batch_size = len(inputs)
  
  for i, input in enumerate(inputs):
    result = []
    if 'lengths' not in kwargs:
      length = input['word_count'] + 1
    else:
      length = kwargs['lengths'][i]
    for dep in range(1, length):
      if 'mask' in kwargs:
        if kwargs['mask'][i][0][dep][0] == 0:
          continue
      head = torch.argmax(arc_attention[i, 0, dep, :length]).item()
      result.append({
        'dep': dep,
        'head': head
      })
      
    if 'dependency' in input:
      key = 'dependency_predict'
    else:
      key = 'dependency'
    input[key] = result
  
  return inputs

@register_run_fn('mtre-sentence-pretrain')
def run_mtre_pretrain(model, inputs, config, **kwargs):
  device = next(model.parameters()).device
  batch_size = len(inputs)
  
  # FIXME
  kwargs['pretrain'] = True

  relations, relation_scores, arc_attention, entities, entity_type_scores = model.run(inputs, **kwargs)

  inputs = parse_greedy(arc_attention, inputs, config, **kwargs)

  # Entity type labeling
  _, top_k = torch.topk(entity_type_scores, config['top-k'], dim=1)
  for i, entity in enumerate(entities):
    if 'label' in entity:
      key = 'label_predict'
    else:
      key = 'label'
    entity[key] = [model.etl_labels[l_i] for l_i in top_k[i].tolist()]

  return inputs

@register_run_fn('mtre-sentence')
def run_mtre(model, inputs, config, **kwargs):
  device = next(model.parameters()).device
  batch_size = len(inputs)
  
  # FIXME
  kwargs['pretrain'] = False

  relations, relation_scores, arc_attention, entities, entity_type_scores = model.run(inputs, **kwargs)

  inputs = parse_greedy(arc_attention, inputs, config, **kwargs)

  # Entity type labeling
  _, top_k = torch.topk(entity_type_scores, config['top-k'], dim=1)
  for i, entity in enumerate(entities):
    if 'label' in entity:
      key = 'label_predict'
    else:
      key = 'label'
    entity[key] = [model.etl_labels[l_i] for l_i in top_k[i].tolist()]
  
  # Relation labeling
  _, relation_max = torch.max(relation_scores, dim=1)
  for relation, relation_id in zip(relations, relation_max):
    if 'label' in relation:
      key = 'label_predict'
    else:
      key = 'label'
    relation[key] = model.re_labels[relation_id]

  return inputs
