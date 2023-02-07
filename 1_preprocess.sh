# Convert from Modu corpus to KMDP
python kmdp/reformat_modu.py

# Convert train/dev/test split into KMDP
python3 kmdp/generate_kmdp.py --src_file "kmdp/corpus/VictorNLP_kor(Modu)_train.json" --dst_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_train.json"
python3 kmdp/generate_kmdp.py --src_file "kmdp/corpus/VictorNLP_kor(Modu)_dev.json" --dst_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_dev.json"
python3 kmdp/generate_kmdp.py --src_file "kmdp/corpus/VictorNLP_kor(Modu)_test.json" --dst_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_test.json"

python3 kmdp/kmdp_labels.py --src_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_train.json" --dst_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_labels.json"

