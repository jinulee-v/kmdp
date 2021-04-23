"""
@module rules
Defines rule format and its instances.

@register_kmdp_rule('rule_name', ['rules', 'to', 'dominate'])
def function(dep_word_phrase, )
"""

from . import register_kmdp_rule

from .lexicals import head2label, label2head, dp_labels

class KMDPRuleBase():
  """
  Base rule structure for KMDP.
  """

  @classmethod
  def generate(cls, dep_wp, dep_wp_i, head_wp, dp_label):
    """
    Generate KMDP from given PoS and DP results.
    Notations:
    나    는                                    먹        었 다
    --                                         --
  dep_wp[dep_wp_i]                    (desired head morph)
    --------                                   --------------
     dep_wp        -(dp_label=='NP_SBJ')->        head_wp
    
    @param dep_wp List[Dict]. dependent word-phrase.
    @param dep_wp_i Integer.
    @param head_wp List[Dict]. head word-phrase.
    @param dp_label String. DP label from dep_wp to head_wp

    @return Union[None, Dict {
      "dep": Int,       # dep_wp[dep_wp_i]["id"]
      "head": Int,      # generated
      "label": String   # generated
    }]
    """
    raise NotImplementedError()
  
  @classmethod
  def recover(cls, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):
    """
    Generate DP from given KMDP relation.
    Notations:
    나    는                                    먹        었 다
    --                                         --
  dep_wp[dep_wp_i]                    head_wp[head_wp_i]
    --------                                   --------------
     dep_wp        -(dp_label=='NP_SBJ')->        head_wp
    
    @param dep_wp List[Dict]. dependent word-phrase.
    @param dep_wp_i Integer.
    @param head_wp List[Dict]. head word-phrase.
    @param head_wp_i Integer.
    @param kmdp_label String.

    @return Union[None, Dict {
      "dep": None,       # == dep_wp  's index in sent['pos']; will be filled by the script
      "head": None,      # == head_wp 's index in sent['pos']; will be filled by the script
      "label": String    # generated
    }]
    """
    raise NotImplementedError()

class KMDPIntraWPRuleBase(KMDPRuleBase):
  """
  Base rule structure for Intra-WP rules.
  In this case, recover() is not needed.
  """

  @classmethod
  def recover(cls, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):
    return None

###############################################################################

@register_kmdp_rule("default_inter", [])
class DefaultInterRule(KMDPRuleBase):
  """
  Default rule that can be applied to Inter-WP dependencies.
  """

  def generate(cls, dep_wp, dep_wp_i, head_wp, dp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] not in heads['all_heads']:
      # If non-head PoS, do nothing.
      return None
    
    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in heads['all_heads']:
        # Intra-WP dependency
        return None
    
    # if dep_wp_i-th morpheme is the last head of dep_wp:
    # Inter-WP dependency
    # Return the last head morpheme from the head_wp.
    for i in reversed(range(len(head_wp))):
      if head_wp[i]['pos_tag'] in heads['all_heads']:
        return {
          'dep': dep_morph['id'],
          'head': head_wp[i]['id'],
          'label': dp_label
        }
    raise ValueError('No head morpheme in head_wp!')
  
  def recover(cls, dep_wp, dep_wp_i, head_wp, head_wp_i, kmdp_label):
    dep_morph = dep_wp[dep_wp_i]
    if dep_morph['pos_tag'] not in heads['all_heads']:
      # If non-head PoS, do nothing.
      return None
    
    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in heads['all_heads']:
        # Intra-WP dependency
        return None
    
    if dep_wp == head_wp:
      return None

    if kmdp_label not in dp_labels:
      raise ValueError('Invalid DP label: {}'.format(kmdp_label))

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
    if dep_morph['pos_tag'] not in heads['all_heads']:
      # If non-head PoS, do nothing.
      return None
    
    # If head PoS, find next head morpheme.
    for i in range(dep_wp_i, len(dep_wp)):
      if dep_wp[i]['pos_tag'] in heads['all_heads']:
        # Intra-WP dependency
        return {
          'dep': dep_morph['id'],
          'head': dep_wp[i]['id'],
          'label': head2label[dep_morph['pos_tag']]
        }
    
    # Not an intra-dependency
    return None

    