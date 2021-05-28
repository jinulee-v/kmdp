class KMDPRuleBase():
  """
  Base rule structure for KMDP.
  """

  def __init_subclass__(cls, *args, **kwargs):
    cls.generate = classmethod(cls.generate)

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

###############################################################################