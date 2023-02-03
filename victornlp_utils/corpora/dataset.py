"""
@module dataset
'VictorNLP Framework' corpus formatting functions.

'@param inputs' shares the following format.
[
  {
    'text': (raw sentence)
    'pos': (pos-tagged format)
    'dependency': [
       {
         dep:
         head:
         label:
       }, ...
     ],
     ...
  },
  ...
]
"""

from torch.utils.data import Dataset

dataset_preprocessors = {}
def register_preprocessors(name):
  def decorator(fn):
    dataset_preprocessors[name] = fn
    return fn
  return decorator

class VictorNLPDataset(Dataset):
  """
  Transparent wrapper for 'List of dictionaries' format. Implements map-style Dataset class.
  """
  
  def __init__(self, inputs, preprocessors=[]):
    """
    Constructor for VictorNLPDataset.
    
      @param inputs List of dictionaries.
      @param preprocessors List of callables. preprocessor_*() or other callable formats that take 'inputs' is accepted.
    """
    for preprocessor in preprocessors:
      inputs = preprocessor(inputs)
    self._data = inputs

  def __getitem__(self, idx):
    return self._data[idx]
  
  def __len__(self):
    return len(self._data)
  
  def collate_fn(batch):
    return batch

class VictorNLPPairDataset(Dataset):
  """
  Special class for Sentence Pair Tasks(NLI, STS, Paraphrase detection, ...)
  """

  def __init__(self, inputs_a, inputs_b, inputs_pairinfo, preprocessors=[]):
    """
    Constructor for VictorNLPDataset.
    
      @param inputs_a List of dictionaries: VictorNLP corpus format.
      @param inputs_b List of dictionaries: VictorNLP corpus format.
      @param inputs_pairinfo List of objects: Whatever data needed to store the information pair. It may be a label(paraphrase/NLI) or a floating point(STS), etc.
      @param preprocessors List of callables. preprocessor_*() or other callable formats that take 'inputs' is accepted.
    """
    # All three parameters should have equal lengths.
    assert len(inputs_a) == len(inputs_b) == len(inputs_pairinfo)

    for preprocessor in preprocessors:
      inputs_a = preprocessor(inputs_a)
      inputs_b = preprocessor(inputs_b)
    self._data_a = inputs_a
    self._data_b = inputs_b
    self._data_pairinfo = inputs_pairinfo

  def __getitem__(self, idx):
    return (self._data_a[idx], self._data_b[idx], self._data_pairinfo[idx])
  
  def __len__(self):
    return len(self._data_a)

  def collate_fn(batch):
    return [pair[0] for pair in batch], [pair[1] for pair in batch], [pair[2] for pair in batch]
  

###############################################################################

@register_preprocessors('word-count')
def preprocessor_WordCount(inputs):
  """
  Adds word_count to inputs.
  """
  for input in inputs:
     assert input['text']
     input['word_count'] = len(input['text'].split())

  return inputs

@register_preprocessors('dependency-parsing')
def preprocessor_DependencyParsing(inputs):
  """
  Checks data integrity for dependency parsing.
  
  @param inputs List of dictionaries.
  
  @return modified 'inputs' for dependency parsing
  """
  for input in inputs:
     assert input['text']
     assert input['pos']
     assert len(input['pos']) == input['word_count']
    
  return inputs
