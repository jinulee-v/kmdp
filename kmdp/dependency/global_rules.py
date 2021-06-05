"""
@module global_rules

Implements two special rules that requires globally scoped scanning.
"""

from .rules_base import *

from .interface import kmdp_generate, KMDPGenerateException
from .lexicals import label2head, head2label, brackets
from . import register_global_rule, kmdp_rules


# Implementation of global rule-based dependency functions.
# All rule functions should use @register_global_rule decorator to be properly accessed.

@register_global_rule('global_recursive_head', [])
class RecursiveHeadRule():
  """
  Recursively runs up the DP tree until find any valid head morpheme.
  About ~0.5% of sentences meet this condition.
  Note: Recovering DP result from this rule is inherently impossible;
        it should be excluded when calculating accuracy.
  """

  def generate(sentence, dep_id, dep_wp_i):
    pos = [[{'id': 0, 'text': '[ROOT]', 'pos_tag': '[ROOT]'}]] + sentence['pos']
    dp = [{}] + sentence['dependency']

    dep_wp = pos[dep_id]
    if dep_wp[dep_wp_i]['pos_tag'] not in label2head['all_heads']:
      return None
    curr_id = dep_id

    first_round = True
    head_id = None
    # Loop until reach [ROOT]
    while True:
      head_id = dp[curr_id]['head']
      head_wp = pos[head_id]
      # If reached [ROOT], break.
      if head_id == 0:
        break
        
      for head_morph in head_wp:
        # If head morpheme exists,
        if head_morph['pos_tag'] in label2head['all_heads']:
          if first_round:
            # Can parse normally with kmdp_rules
            return None
          else:
            for rule in kmdp_rules:
              # Skip global rules
              if 'global_' in rule:
                continue
              dp_label = dp[dep_id]['label']
              result = kmdp_generate(rule, dep_wp, dep_wp_i, head_wp, dp_label)
              # If found rule that applies, return the value
              if result:
                return result
            # If no rules can apply... (this will raise an error in generate_kmdp.py)
            if not result:
              return None
      first_round = False
      curr_id = head_id
    return None


@register_global_rule('global_descriptive_brackets', ['global_scope'])
class DescriptiveBracketsRule():
  """
  Peforms different parsing method for 'Descriptive Brackets':
  Descriptive brackets: Brakets that are attached to preceding word-phrase
    ex.        형태소 기반 분석(상당히 좋은 접근이다)을 채택했다.
    WP-level:              dep    head                   : AP
                                  dep   head             : NP_ADJ
                                         dep     head    : NP_OBJ
    morph-level:         head            dep             : CP
  Modifies dependency of described morpheme(분석) and describing head(다).

  Note: This rule breaks head-finality, be cautious when implementing .
  Note: This rule assumes max(level)=1 for brackets. FIXME
  """

  def generate(sentence, dep_id, dep_wp_i):
    pos = [[{'id': 0, 'text': '[ROOT]', 'pos_tag': '[ROOT]'}]] + sentence['pos']
    dp = [{}] + sentence['dependency']

    # Brackets
    pair_open = {}; pair_close = {}
    for open, close in brackets:
      pair_open[open] = close
      pair_close[close] = open

    dep_wp = pos[dep_id]
    if dep_wp[dep_wp_i]['pos_tag'] not in label2head['all_heads']:
      return None
    for bracket_i, morph in enumerate(dep_wp):
      if bracket_i <= dep_wp_i:
        continue
      if morph['pos_tag'] in label2head['all_heads']:
        return None
      if morph['text'] in list(pair_open.keys()) + list(pair_close.keys()):
        break
      bracket_i = None
    if not bracket_i:
      return None

    # Described morpheme
    if dep_wp_i < len(dep_wp)-1 and dep_wp[bracket_i]['text'] in pair_open:
      close_index = None
      for i, head_wp in enumerate(pos):
        # search only WPs after dep_wp
        if i < dep_id:
          continue
        # if closing bracket is present...
        try:
          close_index = [morph['text'] for morph in head_wp].index(pair_open[dep_wp[bracket_i]['text']])
          break
        except:
          continue
      if close_index is None:
        return None
      
      # Found closing bracket!
      dep_wp = dep_wp[dep_wp_i:dep_wp_i+1]
      if i == dep_id:
        # brackets are in the same wp
        intra_wp_head = False
        for j in range(close_index, len(head_wp)):
          if head_wp[j]['pos_tag'] in label2head['all_heads']:
            # intra-wp dependency
            # 노트/NNG+(/SS+백서/NNG+)/SS+이/VCP+라도/EC
            intra_wp_head = True
            dp_label = head2label[head_wp[j]['pos_tag']]
            head_wp = head_wp[j:]
            break
        if not intra_wp_head:
          # inter-wp dependency
          # 트랜스젠더/NNG+(/SS+성전환자/NNG+)/SS 학생/NNG+들/XSN+은/JX
          head_wp = pos[dp[dep_id]['head']]
          dp_label = dp[dep_id]['label']
        
      else:
        # brackets are in another wp
        head_wp = pos[dp[i]['head']]
        dp_label = dp[i]['label']
        if close_index == 0:
          raise KMDPGenerateException('global_descriptive_brackets', 'Word-phrase starts with a closing bracket; not expected in descriptive_brackets', dep_wp, dep_wp_i, head_wp, dp_label)

      for rule in kmdp_rules:
        # Skip global rules
        if 'global_' in rule:
          continue
        result = kmdp_generate(rule, dep_wp, 0, head_wp, dp_label)
        # If found rule that applies, return the value
        if result:
          return result
      # If no rules can apply... (this will raise an error in generate_kmdp.py)
      if not result:
        return None


    # Describing head
    elif dep_wp_i < len(dep_wp)-1 and dep_wp[bracket_i]['text'] in pair_close:
      open_index = None
      head_wp = None
      for i, head_wp_temp in enumerate(pos):
        # search only WPs after dep_wp
        if i > dep_id:
          break
        # if opening bracket is present...
        try:
          open_index = [morph['text'] for morph in head_wp_temp].index(pair_close[dep_wp[bracket_i]['text']])
          head_wp = head_wp_temp
        except:
          continue
      if open_index is None:
        return None
      
      # Found closing bracket!

      if open_index == 0 or head_wp[open_index-1]['pos_tag'] not in label2head['all_heads']:
        # Not a descriptive bracket
        # 정부 합의안이 (국회에서) 무시당한
        return None
      dep_wp = dep_wp[dep_wp_i:dep_wp_i+1]
      dp_label = head2label[dep_wp[0]['pos_tag']] + '_AJT'
      head_wp = head_wp[:open_index]

      for rule in kmdp_rules:
        # Skip global rules
        if 'global_' in rule:
          continue
        result = kmdp_generate(rule, dep_wp, 0, head_wp, dp_label)
        # If found rule that applies, return the value
        if result:
          return result
      # If no rules can apply... (this will raise an error in generate_kmdp.py)
      if not result:
        return None
    
    # Nothing
    else:
      return None


@register_global_rule('global_scope', [])
class ScopeRule():
  """
  Scope rule. Dependency inside brackets and quotations remain internal.
  """

  def generate(sentence, dep_id, dep_wp_i):
    pos = [[{'id': 0, 'text': '[ROOT]', 'pos_tag': '[ROOT]'}]] + sentence['pos']
    dp = [{}] + sentence['dependency']

    # Brackets
    pair_open = {}; pair_close = {}
    for open, close in brackets:
      pair_open[open] = close
      pair_close[close] = open

    # Index scope with morphh[id]
    if 'scope' not in sentence:
      scope = []
      level = 1
      stack = []
      for wp in pos:
        for morph in wp:
          scope.append(level)
          if morph['text'] in pair_close and len(stack)>0 and stack[-1] == pair_close[morph['text']]:
            del stack[-1]
            level -= 1
          elif morph['text'] in pair_open:
            stack.append(morph['text'])
            level += 1
      if len(stack) == 0:
        if level != 1:
          raise KMDPGenerateException('global_scope', 'incomplete scope', dep_wp, dep_wp_i, head_wp, dp_label)
      sentence['scope'] = scope
    else:
      scope = sentence['scope']
    
    dep_wp = pos[dep_id]
    head_wp = pos[dp[dep_id]['head']]
    dp_label = dp[dep_id]['label']

    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['all_heads']:
        # Intra-WP dependency
        return None
    if scope[head_wp[0]['id']] == scope[head_wp[-1]['id']]:
      return None
    
    modified = False
    if scope[dep_wp[dep_wp_i]['id']] <= scope[head_wp[0]['id']]:
      # infront of the scope
      for i, head_morph in enumerate(head_wp):
        if scope[dep_wp[dep_wp_i]['id']] != scope[head_morph['id']]:
          head_wp_new = head_wp[:i]
          modified = True
          break
    if not modified and scope[dep_wp[dep_wp_i]['id']] >= scope[head_wp[-1]['id']]:
      # infront of the scope
      for i, head_morph in enumerate(head_wp):
        if scope[dep_wp[dep_wp_i]['id']] == scope[head_morph['id']]:
          head_wp_new = head_wp[i:]
          modified = True
          break
    if modified:
      for head_morph in head_wp_new:
        if head_morph['pos_tag'] in label2head['all_heads']:
          head_wp= head_wp_new
          break
    else:
      pass
      # raise KMDPGenerateException('global_scope', 'scope unmatch', dep_wp, dep_wp_i, head_wp, dp_label)
    
    for rule in kmdp_rules:
      # Skip global rules
      if 'global_' in rule:
        continue
      dp_label = dp[dep_id]['label']
      result = kmdp_generate(rule, dep_wp, dep_wp_i, head_wp, dp_label)
      # If found rule that applies, return the value
      if result:
        return result
    # If no rules can apply... (this will raise an error in generate_kmdp.py)
    if not result:
      return None

