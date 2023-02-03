import torch
import torch.nn as n

from . import register_parse_fn

@register_parse_fn('beam')
def parse_beam(parser, inputs, config, **kwargs):
  """
  Beam search implemented in Stack-Pointer Network(Ma, 2018) and Left-To-Right parser(Fernandez-Gonzalez, 2019).
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  @param **kwargs Passed to parser.run().
  
  @return 'inputs' dictionary with parse tree information added.
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs, **kwargs)

  head_initial = config['head_initial']
  head_final = config['head_final']
  allow_projective = config['allow_projective']
  beam_size = config['beam_size']

  for i, input in enumerate(inputs):

    # Arc decision
    beam = [[0]]
    beam_scores = torch.ones(beam_size).to(device) * float('-inf')
    beam_scores[0] = 0.

    arc_att = torch.squeeze(arc_attention[i].clone(), 0)

    if 'lengths' not in kwargs:
      length = input['word_count'] + 1
    else:
      length = kwargs['lengths'][i]
    for dep_id in range(1, length):
      new_scores = beam_scores.unsqueeze(0) + arc_att[dep_id].unsqueeze(1)
      new_scores, indices = torch.sort(new_scores.view(-1), descending=True)
      indices = list(indices.cpu())
      for j, index in enumerate(indices):
        index = index.item()
        indices[j] = (index//beam_size, index%beam_size)

      new_beam = []
      new_beam_scores = torch.ones(beam_size).to(device) * float('-inf')
      for score, (head_id, beam_id) in zip(new_scores, indices):
        if len(new_beam) == beam_size or beam_id >= len(beam):
          break
        if head_id >= length:
          continue

        # Check topological conditions
        if head_initial:
          if head_id >= dep_id:
            continue
        if head_final:
          if (dep_id == length - 1) and (head_id != 0):
            continue
          elif (dep_id < length - 1) and (head_id <= dep_id):
            continue
        if not allow_projective:
          projective = False
          for child, parent in enumerate(beam[beam_id]):
            if parent < dep_id < child or child < dep_id < parent:
              if (head_id < child and head_id < parent) or (head_id > child and head_id > parent):
                projective = True
                break
            if parent < head_id < child or child < head_id < parent:
              if (dep_id < child and dep_id < parent) or (dep_id > child and dep_id > parent):
                projective = True
                break
          if projective:
            continue

        # Add current parse to list
        new_beam_scores[len(new_beam)] = score
        new_beam.append(beam[beam_id]+[head_id])
      # Update beam
      beam = new_beam
      beam_scores = new_beam_scores

    heads = beam[0]

    # Type decision
    types = []
    type_att = type_attention[i]
    for dep_id, head_id in enumerate(heads):
      types.append(torch.argmax(type_att[:, dep_id, head_id]))

    # Update result
    result = []
    for dep_id, (head_id, type_id) in enumerate(zip(heads, types)): # FIXME
      if dep_id == 0:
        continue
      result.append({'dep': dep_id, 'head': head_id, 'label': parser.labels[type_id]})
    if 'dependency' in input:
      key = 'dependency_predict'
    else:
      key = 'dependency'
    input[key] = result

  return inputs