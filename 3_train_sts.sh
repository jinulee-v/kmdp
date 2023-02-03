# Training presets

cd sent_text_sim

python -m victornlp_sts.train \
    --config-file models/Korean_STS_TreeLSTM+GloVe/config.json

python -m victornlp_sts.train \
    --config-file models/Korean_STS_TreeLSTM+KoBERT/config.json

python -m victornlp_sts.train \
    --config-file models/Korean_STS_TreeLSTM+KorBERT/config.json

python -m victornlp_sts.train \
    --config-file models/Korean_STS_TreeLSTM+GloVe_KMDP/config.json

python -m victornlp_sts.train \
    --config-file models/Korean_STS_TreeLSTM+GloVe_KMDP/config.json
