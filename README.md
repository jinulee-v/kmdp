# KMDP: Korean Morpheme-based Dependency Parsing

## Summary

- Generation of the Korean Morpheme-based Dependency Tree corpus from Word-phrase based DP corpora
- Implementation of a `TreeLSTM`-`Semantic Textual Similarity` model (Tai, 2015)
- Implementation of a `C-GCN over Pruned Dependency Tree`-`Relation extraction` model (Zhang, 2018)

### Paper information

이진우, 조혜미, 박수연, 신효필. (2021). 한국어 의존 구문 분석의 분석 단위에 관한 실험적 연구. 제33회 한글 및 한국어 정보처리 학술대회 논문집

초록.
현재 한국어 의존 구문 분석의 표준은 어절 단위로 구문 분석을 수행하는 것이다. 그러나 의존 구문 분석의 분석 단위(어절, 형태소)에 대해서는 현재까지 심도 있는 비교 연구가 진행된 바 없다. 본 연구에서는 의존 구문 분석의 분석 단위가 자연어 처리 분야의 성능에 유의미한 영향을 끼침을 실험적으로 규명한다. STEP 2000과 모두의 말뭉치를 기반으로 구축한 형태소 단위 의존 구문 분석 말뭉치를 사용하여, 의존 구문 분석 기 모델 및 의존 트리를 입력으로 활용하는 문장 의미 유사도 분석(STS) 및 관계 추출(RE) 모델을 학습하였다. 그 결과, KMDP가 기존 어절 단위 구문 분석과 비교하여 의존 구문 분석기의 성능과 응용 분야(STS, RE)의 성능이 모두 유의미하게 향상됨을 확인하였다. 이로써 형태소 단위 의존 구문 분석이 한국어 문법을 표현하는 능력이 우수하며, 문법과 의미를 연결하는 인터페이스로써 높은 활용 가치가 있음을 입증한다.

---

Lee, J., Cho, H., Bock, S. Shin, H. (2021). Empirical Research on the Segmentation Method for Korean Dependency Parsing.  In Proceedings of The 33rd Annual Conference on Human & Cognitive Language Technology. 

Abstract.
The standard tokenization policy of Korean dependency parsing is eojeol(word-phrase), separated by whitespaces. However, no previous studies have attempted to compare two tokenization strategies, word-phrase and morpheme. This empirical study proves that the tokenization strategy may influence the performance of the dependency parsing and its downstream tasks. Using a custom morpheme-based dependency corpus built from STEP 2000 and the Modoo corpus, we trained a dependency parser, and semantic textual similarity(STS)/relation extraction(RE) model that exploits explicit dependency information. Results show that morpheme tokenization leads to marginally improved results in dependency parsing, STS and RE. We state that morpheme based dependency parsing provides better structure for encoding Korean syntax and is a better interface between the syntax and the semantics.


## Prerequisites

1. Clone from the GitHub repository.

```
git clone https://github.com/jinulee-v/kmdp
cd kmdp # Current location is refered as 'root folder' from now on.
```

2. Download corpus file from the NIKL Homepage.

> Corpus files are copyrighted by NIKL, therefore I do not own rights to distribute them.
> Download the file from the NIKL Modu corpus homepage(Dependency & Morphology).
> This project uses only 'literal text dataset'; `NXDP*.json` and `NXMP*.json`. Discard `SXDP` and `SXMP`.

Store the downloaded files within the root folder.

3. Install relevant packages. Using virtual environments for version control(Docker, venv, conda...) is highly recommended.

```
pip install -r requirements.txt
```

## Instructions

To repeat experiments presented within the paper, follow these steps:

### Run `1_preprocess.sh` .
   
This script...
- Converts NIKL data into our own JSON format.
- Converts word-phrase corpora into KMDP corpora.
After the execution, preprocessed corpora(Modu/ModuKMDP, train/dev/test/labels) will be stored in `kmdp/corpus`.
Also, the error log generated while converting word-phrase dependencies to morpheme dependencies will be stored in `kmdp/logs`.

**FAQ**
- This script requires at least 16GB of RAM space.
- Make sure that `NXDP*.json` nad `NXMP*.json` file is in the root directory, and their names are unchanged.

> For more usage, refer to `python kmdp/generate_kmdp.py -h`.

### Run `2_train_dp.sh` .

This script...
- Trains various dependency parcing models with respective configurations.
After the execution, trained models will be found at `dependency_parsing/models/*`.

> For more usage, refer to `dependency_parsing/README.md`.

### Run `3_train_sts.sh` .

This script...
- Trains various semantic text similarity models with respective configurations.
After the execution, trained models will be found at `sem_text_sim/models/*`.

> For more usage, refer to `sem_text_sim/README.md`.

### Run `4_train_re.sh` . (Not prepared yet)

This script...
- Trains various relation extraction models with respective configurations.
After the execution, trained models will be found at `relation_extraction/models/*`.

> For more usage, refer to `relation_extraction/README.md`.
