"""
@module parse
Various parsing functions based on attention scores.

parse_*(parser, inputs, config)
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
"""

import torch
import torch.nn as nn

from numpy import exp

from . import register_parse_fn

corr_conf, corr_conf_cnt = 0, 0
wrong_conf, wrong_conf_cnt = 0, 0
wrong_conf_error = 0

@register_parse_fn('greedy')
def parse_greedy(parser, inputs, config, **kwargs):
  """
  Simple argmax parsing. No topologocal restrictions are applied during parsing, thus may generate improper structures. Use for testing.
  
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  @param **kwargs Passed to parser.run().
  
  @return 'inputs' dictionary with parse tree information added.
  """
  global corr_conf, corr_conf_cnt, wrong_conf, wrong_conf_error, wrong_conf_cnt
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs, **kwargs)
  if 'mask' in kwargs:
    mask = (1 - kwargs['mask']) * 1e6
    arc_attention = arc_attention - mask
    type_attention = type_attention - mask
  
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
      label = parser.labels[torch.argmax(type_attention[i, :, dep, head]).reshape(-1)]
      result.append({
        'dep': dep,
        'head': head,
        'label': label
      })
      
    if 'dependency' in input:
      key = 'dependency_predict'
      for edge in result:
        for golden_edge in input['dependency']:
          if golden_edge['dep'] == edge['dep']:
            if golden_edge['head'] == edge['head']:
              corr_conf_cnt += 1
              corr_conf += exp(arc_attention[i, 0, golden_edge['dep'], golden_edge['head']].item())
            else:
              wrong_conf_cnt += 1
              wrong_conf += exp(arc_attention[i, 0, golden_edge['dep'], golden_edge['head']].item())
              wrong_conf_error += exp(arc_attention[i, 0, golden_edge['dep'], edge['head']].item())
            break
    else:
      key = 'dependency'
    input[key] = result
  
  print(corr_conf, corr_conf_cnt, wrong_conf, wrong_conf_error, wrong_conf_cnt)
  return inputs
