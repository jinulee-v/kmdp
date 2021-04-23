import json


with open('corpus/VictorNLP_kor(Modu)_labels.json', 'r', encoding='UTF-8') as labels_file:
  _labels = json.load(labels_file)

###############################################################################

# PoS label subdivisions

# Noun-like heads
_NP_heads = set([
  'NNG', 'NNP', 'NG', 'NNP', 'NNB', 'NP', 'NR',
  'ETN', 'XSN', 'SL', 'SH', 'SN', 'XR', 'NF'
])

# Verb(Korean adjectives)-like heads
_VP_heads = set([
  'VV', 'VA', 'VX', 'VCP', 'VCN', 'XSV', 'NV'
])

# Adverb-like heads
_AP_heads = set([
  'MAG', 'MAJ', 'XSA'
])

# Adjective-like ehads
_DP_heads = set([
  'MMA', 'MMD', 'MMN', 'ETM'
])

# Independent phrase heads
_IP_heads = set(['IP'])

# Affix heads
_XPN_heads = set(['XPN'])

_all_heads = set(_NP_heads + _VP_heads + _AP_heads + _DP_heads + _IP_heads + XPN_heads)
for head in _all_heads:
  assert head in _labels['pos_labels']

# All possible heads
label2head = {
  'NP': _NP_heads,
  'VP': _VP_heads,
  'AP': _AP_heads,
  'DP': _DP_heads,
  'IP': _IP_heads,
  'XPN': _XPN_heads,
  'all_heads': _all_heads
}
head2label = {}
for label, heads in label2head.items():
  for head in heads:
    head2label[head] = label

###############################################################################

dp_labels = _labels['dp_labels']