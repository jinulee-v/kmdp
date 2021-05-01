
def topological_sort(_kmdp_precedence):
  """
  Orders KMDP classes.
  """
  queue = []
  kmdp_rules = [] # Topologically (reversely) sorted order.
  precedence_left = {alias:len(dominees) for alias, dominees in _kmdp_precedence.items()}
  for alias, edges in precedence_left.items():
    if edges == 0:
      queue.append(alias)

  while queue:
    # current: rule to be freed(all dominees are already in `kmdp_rules`)
    current = queue.pop(0)
    kmdp_rules.append(current)
    
    # update rules that dominate `current`
    for alias in precedence_left.keys():
      if current in _kmdp_precedence[alias]:
        precedence_left[alias] -= 1
      
    # update queue
    for alias, edges in precedence_left.items():
      if edges == 0 and (alias not in queue and alias not in kmdp_rules):
        queue.append(alias)

  if len(kmdp_rules) != len(_kmdp_precedence.keys()):
    for key, value in _kmdp_precedence.items():
      print('{}: {}'.format(key, value))
    raise ValueError("\nPrecedence of rules form a cycle!")
  return list(reversed(kmdp_rules))