# VictorNLP_Utils: Collections of multi-lingual NLP wrappers and training utilities.

## Summary

### VictorNLP_Utils

This module supports embeddings, wrappers and utilities for Neural NLP tasks. In-depth description of resources and their implementation details are recorded right here:

[VictorNLP_Utils Documentation](https://www.notion.so/jinulee/VictorNLP_Utils-2dcdc9c1c8004e688d9af5abfbb2a095)

### VictorNLP Framework

VictorNLP, a PyTorch-based NLP framework, provides intuitive interfaces and carefully designed code modularity for implementing and pipelining NLP modules. For more details, refer to the VictorNLP Specification page.

[VictorNLP Specification](https://www.notion.so/VictorNLP-Specification-e03ed18b4a034e3baa16f17793781a90)

## How-to:

### Download embeddings

#### GloVe embeddings

To use these embeddings, you must download embedding files, VictorNLP-formatted. Navigate to `.../victornlp_utils`, and execute `sudo ./download_glove.sh` to download pretrained GloVes.

> Note: `download_glove.sh` installs `gdown` for downloading large files.

#### `transformers` based embeddings

- bert-base-uncased
- kobert

To use these embeddings, you do not need to download addition files manually. However, you must correctly install dependencies. Look on the warnings!

> Note: We do not support local file loading currently, but will be soon updated.

#### ETRI KorBERT embeddings

This Korean BERT embedding require some work to run on.
- Visit [AI Hub](https://aiopen.etri.re.kr/service_dataset.php), recieve an API key, and the download the model file. Refer to `EmbeddingBERTMorph_kor/readme.txt` for detailed information.
- Move `bert_config.json`, `pytorch_model.bin`, and `vocab.korean_morp.list` to `embedding/data/EmbeddingBERTMorph_kor`.
- Rename `bert_config.json` to `config.json`.