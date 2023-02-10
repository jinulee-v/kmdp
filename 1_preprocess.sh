# Convert from Modu corpus to KMDP
python kmdp/reformat_modu.py

# Convert train/dev/test split into KMDP
python3 kmdp/generate_kmdp.py --src_file "kmdp/corpus/VictorNLP_kor(Modu)_train.json" --dst_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_train.json"
python3 kmdp/generate_kmdp.py --src_file "kmdp/corpus/VictorNLP_kor(Modu)_dev.json" --dst_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_dev.json"
python3 kmdp/generate_kmdp.py --src_file "kmdp/corpus/VictorNLP_kor(Modu)_test.json" --dst_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_test.json"

python3 kmdp/kmdp_labels.py --src_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_train.json" --dst_file "kmdp/corpus/VictorNLP_kor(ModuKMDP)_labels.json"

# Copy files to dependency_parsing corpus folder
cp kmdp/corpus/* dependency_parsing/corpus

# Copy victornlp files if your OS is not Linux
# cp -r victornlp_utils dependency_parsing/victornlp_dp
# cp -r victornlp_utils relation_extraction/victornlp_re
# cp -r victornlp_utils sem_text_sim/victornlp_sts
