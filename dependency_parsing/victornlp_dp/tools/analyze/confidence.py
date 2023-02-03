"""
@module confidence
Tools for analyzing the parser behavior.
"""

from . import register_analysis_fn

from ...model.parse.parse_greedy import corr_conf, corr_conf_cnt, wrong_conf, wrong_conf_cnt


# confidence only works with parse_greedy function
@register_analysis_fn('confidence')
def analyze_confidence(inputs):
  return {
    "total_conf": round((corr_conf+wrong_conf)/(corr_conf_cnt+wrong_conf_cnt)*100, 2),
    "corrrect_conf": round(corr_conf/corr_conf_cnt*100, 2),
    "wrong_conf": round(wrong_conf/wrong_conf_cnt*100, 2)
  }