# Training presets

cd dependency_parsing

# Korean parser models: Word-phrase

# Korean_DeepBiaffParser_KoBERT+PoS
python -m victornlp_dp.train \
  --title Korean_DeepBiaffParser_KoBERT+PoS \
  --language korean \
  --model deep-biaff-parser \
  -e kobert -e pos-wp-kor \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_DeepBiaffParser_KorBERT
python -m victornlp_dp.train \
  --title Korean_DeepBiaffParser_KorBERT \
  --language korean \
  --model deep-biaff-parser \
  -e etri-korbert \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_LeftToRightParser_GloVe+PoS
python -m victornlp_dp.train \
  --title Korean_LeftToRightParser_GloVe+PoS \
  --language korean \
  --model left-to-right-parser \
  -e glove-wp-kor -e pos-wp-kor \
  --loss-fn local-lh \
  --parse-fn beam

# Korean_LeftToRightParser_KoBERT+PoS
python -m victornlp_dp.train \
  --title Korean_LeftToRightParser_KoBERT+PoS \
  --language korean \
  --model left-to-right-parser \
  -e kobert -e pos-wp-kor \
  --loss-fn local-lh \
  --parse-fn beam

# Korean_LeftToRightParser_KorBERT
python -m victornlp_dp.train \
  --title Korean_LeftToRightParser_KorBERT \
  --language korean \
  --model left-to-right-parser \
  -e etri-korbert \
  --loss-fn local-lh \
  --parse-fn beam

###############################################################################

# Korean parser models: KMDP

# Korean_DeepBiaffParser_GloVe
python -m victornlp_dp.train \
  --title KMDP_DeepBiaffParser_GloVe \
  --language korean \
  --model deep-biaff-parser \
  -e glove-morph-kor -e pos-morph-kor \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_DeepBiaffParser_KorBERT
python -m victornlp_dp.train \
  --title KMDP_DeepBiaffParser_KorBERT \
  --language korean \
  --model deep-biaff-parser \
  -e etri-korbert \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_LeftToRightParser_GloVe+PoS
python -m victornlp_dp.train \
  --title KMDP_LeftToRightParser_GloVe+PoS \
  --language korean \
  --model left-to-right-parser \
  -e glove-morph-kor -e pos-morph-kor \
  --loss-fn local-lh \
  --parse-fn mst
  
# Korean_LeftToRightParser_KorBERT
python -m victornlp_dp.train \
  --title KMDP_LeftToRightParser_KorBERT \
  --language korean \
  --model left-to-right-parser \
  -e etri-korbert \
  --loss-fn local-lh \
  --parse-fn mst