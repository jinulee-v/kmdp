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

from . import register_parse_fn

@register_parse_fn('mst')
def parse_MST(parser, inputs, config, **kwargs):
  """
  Wrapper function for Projective / Nonprojective / Head-Initial/Final MST Algorithms.
  Projective : Eisner Algorithm
  Non-Projective : Chu-Liu-Edmonds Algorithm
  Non-Projective + Head-Initial-Final: DP O(n^3)
  Note that Projectiveness and Head-Initial/Finality do not co-occur in most languages.
  
    @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  @param **kwargs Passed to parser.run().
  
  @return 'inputs' dictionary with parse tree information added.
  """
  head_initial = config['head_initial']
  head_final = config['head_final']
  allow_projective = config['allow_projective']
  beam_size = config['beam_size']
  
  if allow_projective:
    if head_initial or head_final:
      raise "MST for projective & head-initial/final trees are not supported."
    else:
      inputs = _parse_MST_Eisner(parser, inputs, config, **kwargs)
  else:
    if head_initial or head_final:
      inputs = _parse_MST_DP(parser, inputs, config, **kwargs)
    else:
      inputs = _parse_MST_ChuLiuEdmonds(parser, inputs, config, **kwargs)
  
  return inputs

####################

def _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
  """
  Backtracking step in Eisner's algorithm.
  - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
  an end position, and a direction flag (0 means left, 1 means right). This array contains
  the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
  - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
  an end position, and a direction flag (0 means left, 1 means right). This array contains
  the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
  - s is the current start of the span
  - t is the current end of the span
  - direction is 0 (left attachment) or 1 (right attachment)
  - complete is 1 if the current span is complete, and 0 otherwise
  - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
  head of each word.
  """
  if s == t:
    return
  if complete:
    r = complete_backtrack[s][t][direction]
    if direction == 0:
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
      return
    else:
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
      return
  else:
    r = incomplete_backtrack[s][t][direction]
    if direction == 0:
      heads[s] = t
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
      return
    else:
      heads[t] = s
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
      _backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
      return

def _parse_MST_Eisner(parser, inputs, config, **kwargs,):
  """
  Implementation of Eisner Algorithm(Eisner, 1996) which is capable of generating non-projective MST for a sequence.
  
  Code citation: Modified from https://github.com/daandouwe/perceptron-dependency-parser
  - Transplant from NumPy to PyTorch
  - Modified input / output details to fit VictorNLP format
  
    @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  
  @return 'inputs' dictionary with parse tree information added.
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs, **kwargs)
  
  for i, input in enumerate(inputs):
    if 'lengths' not in kwargs:
      num_words = input['word_count']
    else:
      num_words = kwargs['lengths'][i] -1
    scores = arc_attention[i, 0, :num_words+1, :num_words+1].transpose(0, 1)

    # Initialize CKY table.
    complete = torch.zeros([num_words+1, num_words+1, 2]).to(device)  # s, t, direction (right=1).
    incomplete = torch.zeros([num_words+1, num_words+1, 2]).to(device)  # s, t, direction (right=1).
    complete_backtrack = -torch.ones([num_words+1, num_words+1, 2], dtype=torch.long)  # s, t, direction (right=1).
    incomplete_backtrack = -torch.ones([num_words+1, num_words+1, 2], dtype=torch.long)  # s, t, direction (right=1).

    incomplete[0, :, 0] -= float("inf")

    # Loop from smaller items to larger items.
    for k in range(1, num_words+1):
      for s in range(num_words-k+1):
        t = s + k

        # First, create incomplete items.
        # left tree
        incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]
        incomplete[s, t, 0] = torch.max(incomplete_vals0)
        incomplete_backtrack[s, t, 0] = (s + torch.argmax(incomplete_vals0))
        # right tree
        incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t]
        incomplete[s, t, 1] = torch.max(incomplete_vals1)
        incomplete_backtrack[s, t, 1] = (s + torch.argmax(incomplete_vals1)).cpu()

        # Second, create complete items.
        # left tree
        complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
        complete[s, t, 0] = torch.max(complete_vals0)
        complete_backtrack[s, t, 0] = (s + torch.argmax(complete_vals0)).cpu()
        # right tree
        complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
        complete[s, t, 1] = torch.max(complete_vals1)
        complete_backtrack[s, t, 1] = (s + 1 + torch.argmax(complete_vals1)).cpu()

    value = complete[0][num_words][1]
    heads = -torch.ones(num_words + 1, dtype=torch.long)
    _backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, num_words, 1, 1, heads)

    value_proj = 0.0 # Total log score
    for m in range(1, num_words+1):
      h = heads[m]
      value_proj += scores[h, m]

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

####################

def _parse_MST_ChuLiuEdmonds(parser, inputs, config, **kwargs):
  """
  Implementation of Chu-Liu(Edmonds) Algorithm(Chu&Liu, 1965; Edmonds, 1967) which is capable of generating projective MST for a sequence.
  
  Code citation: Modified from https://github.com/danifg/Left2Right-Pointer-Parser/blob/master/neuronlp2/tasks/parser.py
  - Transplant several code lines from NumPy  to PyTorch
  - Modified input / output details to fit VictorNLP format
  
    @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
  @param **kwargs Passed to parser.run().
  
  @return 'inputs' dictionary with parse tree information added.
  """
  
  def _find_cycle(par):
    added = torch.zeros(length, dtype=torch.bool)
    added[0] = True
    cycle = set()
    findcycle = False
    for i in range(1, length):
      if findcycle:
        break

      if added[i] or not curr_nodes[i]:
        continue
 
      # init cycle
      tmp_cycle = set()
      tmp_cycle.add(i)
      added[i] = True
      findcycle = True
      l = i

      while par[l] not in tmp_cycle:
        l = par[l]
        if added[l]:
          findcycle = False
          break
        added[l] = True
        tmp_cycle.add(l)

      if findcycle:
        lorg = l
        cycle.add(lorg)
        l = par[lorg]
        while l != lorg:
          cycle.add(l)
          l = par[l]
        break

    return findcycle, cycle

  def _chuLiuEdmonds():
    par = torch.zeros(length, dtype=torch.long)
    # create best graph
    par[0] = -1
    for i in range(1, length):
      # only interested at current nodes
      if curr_nodes[i]:
        max_score = score_matrix[0,i]
        par[i] = 0
        for j in range(1, length):
          if j == i or not curr_nodes[j]:
            continue

          new_score = score_matrix[j, i]
          if new_score > max_score:
            max_score = new_score
            par[i] = j

    # find a cycle
    findcycle, cycle = _find_cycle(par)
    # no cycles, get all edges and return them.
    if not findcycle:
      final_edges[0] = -1
      for i in range(1, length):
        if not curr_nodes[i]:
          continue

        pr = oldI[par[i], i]
        ch = oldO[par[i], i]
        final_edges[ch] = pr
      return

    cyc_len = len(cycle)
    cyc_weight = 0.0
    cyc_nodes = torch.zeros(cyc_len, dtype=torch.long)
    id = 0
    for cyc_node in cycle:
      cyc_nodes[id] = cyc_node
      id += 1
      cyc_weight += score_matrix[par[cyc_node], cyc_node]

    rep = cyc_nodes[0]
    for i in range(length):
      if not curr_nodes[i] or i in cycle:
        continue

      max1 = float("-inf")
      wh1 = -1
      max2 = float("-inf")
      wh2 = -1

      for j in range(cyc_len):
        j1 = cyc_nodes[j]
        if score_matrix[j1, i] > max1:
          max1 = score_matrix[j1, i]
          wh1 = j1

        scr = cyc_weight + score_matrix[i, j1] - score_matrix[par[j1], j1]

        if scr > max2:
          max2 = scr
          wh2 = j1

      score_matrix[rep, i] = max1
      oldI[rep, i] = oldI[wh1, i]
      oldO[rep, i] = oldO[wh1, i]
      score_matrix[i, rep] = max2
      oldO[i, rep] = oldO[i, wh2]
      oldI[i, rep] = oldI[i, wh2]

    rep_cons = []
    for i in range(cyc_len):
      rep_cons.append(set())
      cyc_node = cyc_nodes[i]
      for cc in reps[cyc_node]:
        rep_cons[i].add(cc)

    for i in range(1, cyc_len):
      cyc_node = cyc_nodes[i]
      curr_nodes[cyc_node] = False
      for cc in reps[cyc_node]:
        reps[rep].add(cc)

    _chuLiuEdmonds()

    # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
    found = False
    wh = -1
    for i in range(cyc_len):
      for repc in rep_cons[i]:
        if repc in final_edges:
          wh = cyc_nodes[i]
          found = True
          break
      if found:
        break

    l = par[wh]
    while l != wh:
      ch = oldO[par[l], l]
      pr = oldI[par[l], l]
      final_edges[ch.item()] = pr.item()
      l = par[l]
  
  ####################

  batch_size = len(inputs)
  
  arc_attention, type_attention = parser.run(inputs, **kwargs)
  
  for i, input in enumerate(inputs):
    if 'lengths' not in kwargs:
      length = input['word_count'] + 1
    else:
      length = kwargs['lengths'][i]
    
    # get original score matrix
    arc_att = arc_attention[i, 0, :length, :length].transpose(0,1).cpu()
    # initialize score matrix to original score matrix
    score_matrix = arc_att.clone().detach()

    oldI = torch.zeros((length, length), dtype=torch.long)
    oldO = torch.zeros((length, length), dtype=torch.long)
    curr_nodes = torch.zeros(length, dtype=torch.bool)
    reps = []

    for s in range(length):
      arc_att[s, s] = float("-inf")
      score_matrix[s, s] = float("-inf")
      reps.append(set())
      if 'mask' in kwargs:
        if s == 0 or kwargs['mask'][i, 0, s, 0] == 1:
          curr_nodes[s] = True
        else:
          curr_nodes[s] = False
          continue
      else:
        curr_nodes[s] = True
        score_matrix[s, :] = float("-inf")
        score_matrix[:, s] = float("-inf")
      reps[s].add(s)
      for t in range(s + 1, length):
        oldI[s, t] = s
        oldO[s, t] = t

        oldI[t, s] = t
        oldO[t, s] = s

    final_edges = dict()
    _chuLiuEdmonds()

    heads = [0] * length
    for ch, pr in final_edges.items():
      heads[ch] = int(pr)
    
    # Type decision
    types = []
    type_att = type_attention[i]
    for dep_id, head_id in enumerate(heads):
      types.append(torch.argmax(type_att[:, dep_id, head_id]).item())

    # Update result
    result = []
    for dep_id, (head_id, type_id) in enumerate(zip(heads, types)): # FIXME
      if dep_id == 0:
        continue
      if 'mask' in kwargs:
        if kwargs['mask'][i, 0, dep_id, 0] == 0:
          continue
      result.append({'dep': dep_id, 'head': head_id, 'label': parser.labels[type_id]})
    if 'dependency' in input:
      key = 'dependency_predict'
    else:
      key = 'dependency'
    input[key] = result
  
  return inputs
