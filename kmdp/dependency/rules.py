"""
@module rules
Defines rule format and its instances.

@register_kmdp_rule('rule_name', ['rules', 'to', 'dominate'])
def function(dep_word_phrase, )
"""

from . import register_kmdp_rule

class KMDPRuleBase():

  def __init_subclass__(cls, *args, **kwargs):
    # automatically register subclasses
    return register_kmdp_rule(cls)

  @classmethod
  def generate(dep_wp, dep_wp_i, head_wp, dp_label):
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

    @return Dict {
      "dep": Int,       # dep_wp[dep_wp_i]["id"]
      "head": Int,      # generated
      "label": String   # generated
    }
    """
    raise NotImplementedError()
  
  @classmethod
  def recover(dep_wp, dep_wp_i, head_wp, head_wp_i):
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

    @return Dict {
      "dep": None,       # null
      "head": None,      # null
      "label": String   # generated
    }
    """
    raise NotImplementedError()