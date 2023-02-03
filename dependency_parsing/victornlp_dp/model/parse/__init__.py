"""
@module parse
Various parsing functions based on attention scores.

parse_*(parser, inputs, config)
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
"""

dp_parse_fn = {}
def register_parse_fn(name):
  def decorator(fn):
    dp_parse_fn[name] = fn
    return fn
  return decorator

from .parse_greedy import parse_greedy
from .parse_beam import parse_beam
from .parse_MST import parse_MST