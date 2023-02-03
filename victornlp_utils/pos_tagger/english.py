"""
@module pos_tagger/english

Implements VictorNLP_style wrapper for NLTK PoS tagger.
"""

import nltk

from . import register_pos_tagger

nltk.download('averaged_perceptron_tagger')

@register_pos_tagger('english')
def wrapper_nltk(inputs):
  """
  English pos tagger. Uses NLTK, distributed by Stanford NLP Group.
  
  @param inputs List of dictionaries.
  @return inputs Updated inputs.
  """
  for input in inputs:
    assert 'text' in input
    assert 'word_count' in input

    if 'pos' in input:
      continue
    
    words = input['text'].split()
    assert len(words) == input['word_count']
    
    pos_tagged = nltk.pos_tag(words)
    word_i = 1
    pos = []
    for morph_text, morph_tag in pos_tagged:
      pos.append({
        'id': word_i,
        'text': morph_text,
        'pos_tag': morph_tag
      })
      word_i += 1
    assert len(pos) == input['word_count']
    input['pos'] = pos

  return inputs