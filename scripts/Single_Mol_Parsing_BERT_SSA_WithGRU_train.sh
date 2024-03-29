#!/usr/bin/env bash
data_dir='../TSTRL/Data'
train_mol_file="${data_dir}/Molweni/train.json"
eval_mol_file="${data_dir}/Molweni/dev.json"
test_mol_file="${data_dir}/Molweni/test.json"
train_hu_file="${data_dir}/Hu_Dataset/Hu_Link_Dir/train.json"
train_ou5_file="${data_dir}/Ou_Dataset/Ou5_Link_Dir/train.json"
train_ou10_file="${data_dir}/Ou_Dataset/Ou10_Link_Dir/train.json"
train_ou15_file="${data_dir}/Ou_Dataset/Ou15_Link_Dir/train.json"
train_hu_rs_file="${data_dir}/Hu_Dataset/train.json"
train_ou5_rs_file="${data_dir}/Ou_Dataset/5_train.json"
train_ou10_rs_file="${data_dir}/Ou_Dataset/10_train.json"
train_ou15_rs_file="${data_dir}/Ou_Dataset/15_train.json"
eval_hu_rs_file="${data_dir}/Hu_Dataset/dev.json"
eval_ou5_rs_file="${data_dir}/Ou_Dataset/5_dev.json"
eval_ou10_rs_file="${data_dir}/Ou_Dataset/10_dev.json"
eval_ou15_rs_file="${data_dir}/Ou_Dataset/15_dev.json"
dataset_dir="../TSTRL/dataset_dir"
model_dir="./model_dir_mol_parsing"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
CUDA_VISIBLE_DEVICES=${GPU}  nohup python -u main.py \
                                    --train_mol_file=$train_mol_file \
                                    --eval_mol_file=$eval_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --train_hu_file=$train_hu_file \
                                    --train_ou5_file=$train_ou5_file \
                                    --train_ou10_file=$train_ou10_file \
                                    --train_ou15_file=$train_ou15_file \
                                    --train_hu_rs_file=$train_hu_rs_file \
                                    --train_ou5_rs_file=$train_ou5_rs_file \
                                    --train_ou10_rs_file=$train_ou10_rs_file \
                                    --train_ou15_rs_file=$train_ou15_rs_file \
                                    --eval_hu_rs_file=$eval_hu_rs_file \
                                    --eval_ou_rs_len5_file=$eval_ou5_rs_file \
                                    --eval_ou_rs_len10_file=$eval_ou10_rs_file \
                                    --eval_ou_rs_len15_file=$eval_ou15_rs_file \
                                    --hu_pool_size 100 \
                                    --hu_batch_size 1024 \
                                    --dataset_dir=$dataset_dir  \
                                    --do_train \
                                    --TST_epoches 10 \
                                    --with_GRU \
                                    --ST_model_path "${model_dir}/mol_parsing_bert_ssa_withGRU" \
                                    --TST_model_path "${model_dir}/mol_parsing_bert_ssa_withGRU" \
                                    --seed 65534 > ./logs/mol_parsing_bert_ssa_withGRU.log 2>&1 &