import json


with open('corpus/VictorNLP_kor(Modu)_labels.json', 'r', encoding='UTF-8') as labels_file:
  _labels = json.load(labels_file)

###############################################################################

# PoS label subdivisions

# Noun-like heads
_NP_heads = [
  'NNG', 'NNP', 'NNG', 'NNP', 'NNB', 'NP', 'NR',
  'ETN', 'XSN', 'SL', 'SH', 'SN', 'XR', 'NF'
]

# Verb(Korean adjectives)-like heads
_VP_heads = [
  'VV', 'VA', 'VX', 'VCP', 'VCN', 'XSV', 'XSA', 'NV'
]

# Tense-suffix heads
_TP_heads = ['EP']

# Tense-suffix heads
_CP_heads = ['EF', 'EC']

# Adverb-like heads
_AP_heads = [
  'MAG', 'MAJ'
]

# Adjective-like ehads
_DP_heads = [
  'MMA', 'MMD', 'MMN', 'ETM'
]

# Independent phrase heads
_IP_heads = ['IC']

# Affix heads
_XPN_heads = ['XPN']

# Root heads
_ROOT_heads = ['[ROOT]']

_all_heads = _NP_heads + _VP_heads + _TP_heads + _CP_heads + _AP_heads + _DP_heads + _IP_heads + _XPN_heads + _ROOT_heads
for head in _all_heads:
  assert head in _labels['pos_labels'] + ['[ROOT]']

# All possible heads
label2head = {
  'NP': _NP_heads,
  'VP': _VP_heads,
  'TP': _TP_heads,
  'CP': _CP_heads,
  'AP': _AP_heads,
  'DP': _DP_heads,
  'IP': _IP_heads,
  'XPN': _XPN_heads,
  'all_heads': _all_heads
}
head2label = {}
for label, heads in label2head.items():
  if label == 'all_heads':
    continue
  for head in heads:
    head2label[head] = label

###############################################################################

dp_labels = _labels['dp_labels']

###############################################################################

# SH Adjectives

SH_adjectives = ['故', '新']

# Brackets

brackets = ['()', '{}', '[]', '<>', '’’', '““']
