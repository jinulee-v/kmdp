"""
@module rules
Defines rule format and its instances.

@register_kmdp_rule('rule_name', ['rules', 'to', 'dominate'])
def function(dep_word_phrase, )
"""

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
    for i in reversed(range(len(head_wp))):
      if head_wp[i]['pos_tag'] in label2head['all_heads']:
        return {
          'dep': dep_morph['id'],
          'head': head_wp[i]['id'],
          'label': head2label[dep_morph['pos_tag']] + ('_' + dp_label.split('_')[-1] if '_' in dp_label else '')
        }
    raise KMDPGenerateException('default_inter', 'No head morpheme in head_wp', dep_wp, dep_wp_i, head_wp, dp_label)
  
  def recover(cls, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] not in label2head['all_heads']:
      # If non-head PoS, do nothing.
      return None
    
    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['all_heads']:
        # Intra-WP dependency
        return None

    if kmdp_label not in dp_labels:
      raise KMDPRecoverException('default_inter', 'Invalid DP label: {}'.format(kmdp_label), dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label)

    return {
      'dep': None,
      'head': None,
      'label': kmdp_label
    }


@register_kmdp_rule("default_intra", [])
class DefaultIntraRule(KMDPIntraWPRuleBase):
  """
  Default rule that can be applied to Intra-WP dependencies.
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
  
  def recover(cls, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['text'] not in SH_adjectives:
      # If not adjective-like chinese character, do nothing.
      return None
    
    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['all_heads']:
        # Intra-WP dependency
        return {
          'dep': None,
          'head': None,
          'label': 'XPN'
        }

    if kmdp_label not in dp_labels:
      raise KMDPRecoverException('default_inter', 'Invalid DP label: {}'.format(kmdp_label), dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label)

    return {
      'dep': None,
      'head': None,
      'label': 'NP'
    }


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
    
    for i in range(len(head_wp)):
      if head_wp[i]['pos_tag'] in label2head['VP']:
        return {
          'dep': dep_morph['id'],
          'head': head_wp[i]['id'],
          'label': dp_label
        }
    
    return None
  
  def recover(cls, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] not in label2head['all_heads']:
      # If not adjective-like chinese character, do nothing.
      return None

    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i+1, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in label2head['all_heads']:
        # Intra-WP dependency
        return None
    
    if head_wp[head_wp_i]['pos_tag'] in label2head['VP']:
      return {
        'dep': None,
        'head': None,
        'label': kmdp_label
      }
    