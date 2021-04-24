# KMDP: Korean Morpheme-based Dependency Parsing

## Summary

TODO

## Installation

1. Clone from the GitHub repository.

```
git clone https://github.com/jinulee-v/kmdp
```

2. Download corpus file.

`./corpus/` should look like:
- VictorNLP_kor(Modu)_train.json
- VictorNLP_kor(Modu)_dev.json
- VictorNLP_kor(Modu)_test.json
- VictorNLP_kor(Modu)_labels.json

> Corpus files are copyrighted by NIKL, therefore I do not own rights to distribute them.

## How-to:

### Generate KMDP results from DP and PoS-tagging results

1. Generate full results in JSON format

```
python kmdp/generate_kmdp.py --src-file corpus/VictorNLP_kor(Modu)_test.json --dst-file result.json
```

2. To see simplified format in command line

```
python kmdp/generate_kmdp.py --src-file corpus/VictorNLP_kor(Modu)_test.json --simple
```

> For more usage, refer to `python kmdp/generate_kmdp.py -h`.