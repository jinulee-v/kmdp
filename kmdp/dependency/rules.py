"""
@module rules
Defines rule format and its instances.

@register_kmdp_rule('rule_name', ['rules', 'to', 'dominate'])
def function(dep_word_phrase, )
"""

import re

from .rules_base import *
from . import register_kmdp_rule

from .lexicals import *
from .interface import KMDPGenerateException

@register_kmdp_rule("default_inter", ['default_intra'])
class DefaultInterRule(KMDPRuleBase):
  """
  Default rule that can be applied to Inter-WP dependencies.
  """

  def generate(cls, dep_wp, dep_wp_i, head_wp, dp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] not in label2head['all_heads']:
      # If non-head PoS, do nothing.
      return None
    
    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['all_heads']:
        # Intra-WP dependency
        return None
    
    # if dep_wp_i-th morpheme is the last head of dep_wp:
    # Inter-WP dependency
    # Return the last head morpheme from the head_wp.
    brackets_opener = ''.join([set[0] for set in brackets])
    brackets_closer = ''.join([set[1] for set in brackets])
    
    head_wp_i = None
    head_bracket_wp_i = None
    inside_bracket = False
    for i in reversed(range(len(head_wp))):
      if head_wp[i]['pos_tag'] in label2head['all_heads']:
        if inside_bracket and not head_bracket_wp_i:
          head_bracket_wp_i = i
        elif not inside_bracket and not head_wp_i:
          head_wp_i = i
          break
      elif head_wp[i]['text'] in brackets_closer:
        inside_bracket = True
      elif head_wp[i]['text'] in brackets_opener:
        inside_bracket = False
    if head_wp_i is None:
      head_wp_i = head_bracket_wp_i

    if head_wp_i is None:
      raise KMDPGenerateException('default_inter', 'No head morpheme in head_wp; or head_wp starts with brackets', dep_wp, dep_wp_i, head_wp, dp_label)
    if head_wp[head_wp_i]['pos_tag'] in label2head['all_heads']:
      return {
        'dep': dep_morph['id'],
        'head': head_wp[head_wp_i]['id'],
        'label': head2label[dep_morph['pos_tag']] + ('_' + dp_label.split('_')[-1] if '_' in dp_label else '')
      }
    raise KMDPGenerateException('default_inter', 'No head morpheme in head_wp', dep_wp, dep_wp_i, head_wp, dp_label)
  

@register_kmdp_rule("default_intra", [])
class DefaultIntraRule(KMDPRuleBase):
  """
  Default rule that can be applied to Intra-WP dependencies.
  """

  @classmethod
  def is_intra_head(cls, morph):
    for non_head in non_intra_head_conditions:
      if morph['pos_tag'] == non_head['pos_tag'] and re.match(non_head['text'], morph['text']):
        return False
    return True

  def generate(cls, dep_wp, dep_wp_i, head_wp, dp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] not in label2head['all_heads']:
      # If non-head PoS, do nothing.
      return None
    
    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['all_heads'] and cls.is_intra_head(dep_wp[i]):
        # Intra-WP dependency
        return {
          'dep': dep_morph['id'],
          'head': dep_wp[i]['id'],
          'label': head2label[dep_morph['pos_tag']]
        }
    
    # Not an intra-dependency
    return None


@register_kmdp_rule("adjective_SH", ['default_inter', 'default_intra'])
class AdjectiveSHRule(KMDPRuleBase):
  """
  Rules for adjective-like Chinese characters.
  For default, Chinese characters(SH) are parsed as nouns.

  - '故/SH' = 'dead'
  - '新/SH' = 'new'
  """

  def generate(cls, dep_wp, dep_wp_i, head_wp, dp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['text'] not in SH_adjectives:
      # If not adjective-like chinese character, do nothing.
      return None
    
    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['NP']:
        # Intra-WP dependency
        return {
          'dep': dep_morph['id'],
          'head': dep_wp[i]['id'],
          'label': 'DP'
        }
    
    # if dep_wp_i-th morpheme is the last head of dep_wp:
    # Inter-WP dependency
    # Return the last head morpheme from the head_wp.
    for i in reversed(range(len(head_wp))):
      if head_wp[i]['pos_tag'] in label2head['all_heads']:
        return {
          'dep': dep_morph['id'],
          'head': head_wp[i]['id'],
          'label': 'DP' + ('_' + dp_label.split('_')[-1] if '_' in dp_label else '')
        }
    raise KMDPGenerateException('adjective_SH', 'Unexpected reach of end', dep_wp, dep_wp_i, head_wp, dp_label)


@register_kmdp_rule("VP_arguments", ['default_inter'])
class VPArgumentsRule(KMDPRuleBase):
  """
  Rules for Inter-WP dependencies where head is VP.
  """

  def generate(cls, dep_wp, dep_wp_i, head_wp, dp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] not in label2head['all_heads']:
      return None

    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['all_heads']:
        # Intra-WP dependency
        return None
    
    head_wp_i = None
    head_wp_i_VX = None
    for i in range(len(head_wp)):
      if head_wp[i]['pos_tag'] in label2head['VP'] and head_wp[i]['pos_tag'] not in ['VX']:
        if head_wp_i:
          raise KMDPGenerateException('VP_arguments', 'Multiple VP heads in one word-phrase', dep_wp, dep_wp_i, head_wp, dp_label)
        head_wp_i = i
      elif head_wp[i]['pos_tag'] in ['VX']:
        head_wp_i_VX = i
        break
    if head_wp_i is None:
      head_wp_i = head_wp_i_VX
    if head_wp_i is not None:
      return {
        'dep': dep_morph['id'],
        'head': head_wp[head_wp_i]['id'],
        'label': head2label[dep_morph['pos_tag']] + ('_' + dp_label.split('_')[-1] if '_' in dp_label else '')
      }
    
    return None


@register_kmdp_rule('NP_adjunct', ['VP_arguments'])
class NPAdjuctRule(KMDPRuleBase):
  """
  Rules for Inter-WP dependencies where head is DP or NP.
  DPs and NPs must bind to NPs.
  """

  def generate(cls, dep_wp, dep_wp_i, head_wp, dp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] not in label2head['DP'] + label2head['NP']:
      return None
    elif dep_morph['pos_tag'] in label2head['NP']:
      if dp_label not in ['NP', 'NP_CNJ', 'NP_MOD']:
        return None

    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['all_heads']:
        # Intra-WP dependency
        return None
    
    # DP/NP -> NP adjunct
    head_wp_i = None
    head_found = False
    for i in range(len(head_wp)):
      if head_wp[i]['pos_tag'] in label2head['NP'] and not \
        (head_wp[i]['pos_tag'] == 'XSN' and head_wp[i]['text'] == '들' or \
        head_wp[i]['pos_tag'] =='ETN'):
        if head_wp_i is not None and head_found:
          raise KMDPGenerateException('NP_adjunct', 'Two or more NP heads', dep_wp, dep_wp_i, head_wp, dp_label)
        if not head_found:
          head_wp_i = i
      elif head_wp[i]['pos_tag'] == 'XPN':
        continue
      elif head_wp[i]['pos_tag'].startswith('S') and head_wp[i]['pos_tag'] not in label2head['all_heads']:
        break
      elif head_wp_i is not None and head_wp[i]['pos_tag'] not in label2head['NP']:
        head_found = True
    if head_wp_i is not None:    
      return {
        'dep': dep_morph['id'],
        'head': head_wp[head_wp_i]['id'],
        'label': head2label[dep_morph['pos_tag']] + ('_' + dp_label.split('_')[-1] if '_' in dp_label else '')
      }
      
    return None


@register_kmdp_rule('EPH', ['default_intra'])
class EPHRule(KMDPRuleBase):
  """
  Rules for Polite EP(-si-; tag EPH originates from KKma PoS tagset).
  EPH's head is its preceding verb.
  """

  def generate(cls, dep_wp, dep_wp_i, head_wp, dp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] != 'EP' or dep_morph['text'] not in ['시', '으시']:
      return None
    
    if dep_wp_i > 0 and dep_wp[dep_wp_i-1]['pos_tag'] in label2head['VP']:
      return {
        'dep': dep_morph['id'],
        'head': dep_wp[dep_wp_i-1]['id'],
        'label': 'POL'
      }
      
    raise KMDPGenerateException('EPH', 'Cannot find preceding verb for EPH', dep_wp, dep_wp_i, head_wp, dp_label)
