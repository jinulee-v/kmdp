"""
@module generate_kmdp
Generates KMDP from DP & PoS tagging results.
"""

from dependency import kmdp_rules, kmdp_generate

def generate_rule_based_KMDP(sentence):
  """
  @brief generate KMDP of the sentence.

  sentence = {
    'text': (raw text),
    'word_count': (N of word-phrases) /*== len(text.split())*/ ,
    'pos': [
      [
        {
          'id': (Nst morpheme in sentence, starts from 1)
          'text': (raw form of the morpheme),
          'pos_tag': (PoS tag)
        }, ... 
      ], ...
    ] /* len(pos) == word_count */ ,
    'dependency': [
      {
        'dep': (dependent is Nth word-phrase)
        'head': (head is Nth word-phrase)
        'label': (dependency label)
      }
    ] /*len(dependency) = word_count*/
  }
  """
  dp = sentence['dependency']
  pos = sentence['pos']
  kmdp = []

  for i, (arc, dep_wp) in enumerate(zip(dp, pos), 1):
    assert i == arc['dep']
    head_wp = pos[arc['head'] - 1] # dp indexing starts from 1
    
    for j in range(len(dep_wp)):
      # for each morpheme,

      for rule in kmdp_rules:
        kmdp_arc = kmdp_generate(rule, dep_wp, j, head_wp, arc['label'])
        if kmdp_arc:
          # if any valid result,
          break
      
      if not kmdp_arc:
        # Only non-head morpheme reach here
        # (default_* rules filter all heads)
        continue
        
      # If any valid result, add to list.
      kmdp.append(kmdp_arc)