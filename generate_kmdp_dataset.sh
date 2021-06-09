$DST_DIR = ~/projects/victornlp_dp/corpus
python3 kmdp/generate_kmdp.py --src-file corpus/VictorNLP_kor(Modu)_train.json --dst-file $DST_DIR/VictorNLP_kor(ModuKMDP)_train.json
python3 kmdp/generate_kmdp.py --src-file corpus/VictorNLP_kor(Modu)_dev.json --dst-file $DST_DIR/VictorNLP_kor(ModuKMDP)_dev.json
python3 kmdp/generate_kmdp.py --src-file corpus/VictorNLP_kor(Modu)_test.json --dst-file $DST_DIR/VictorNLP_kor(ModuKMDP)_test.json

python3 kmdp/kmdp_labels.py --src-file $DST_DIR/VictorNLP_kor(ModuKMDP)_train.json --dst-file $DST_DIR/VictorNLP_kor(ModuKMDP)_labels.json