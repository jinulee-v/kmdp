SRC_DIR=$1

# Convert from Modu corpus to KMDP
python kmdp/reformat_modu.py ${SRC_DIR} 

# Convert train/dev/test split into KMDP
python3 kmdp/generate_kmdp.py --src-file "kmdp/corpus/VictorNLP_kor(Modu)_train.json" --dst-file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_train.json"
python3 kmdp/generate_kmdp.py --src-file "kmdp/corpus/VictorNLP_kor(Modu)_dev.json" --dst-file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_dev.json"
python3 kmdp/generate_kmdp.py --src-file "kmdp/corpus/VictorNLP_kor(Modu)_test.json" --dst-file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_test.json"

python3 kmdp/kmdp_labels.py --src-file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_train.json" --dst-file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_labels.json"

