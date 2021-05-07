class KMDPRuleBase():
  """
  Base rule structure for KMDP.
  """

  def __init_subclass__(cls, *args, **kwargs):
    cls.generate = classmethod(cls.generate)
    cls.recover = classmethod(cls.recover)

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