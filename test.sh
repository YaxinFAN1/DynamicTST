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
eval_hu_rs_file="${data_dir}/Hu_Dataset/test.json"
eval_ou5_rs_file="${data_dir}/Ou_Dataset/5_test.json"
eval_ou10_rs_file="${data_dir}/Ou_Dataset/10_test.json"
eval_ou15_rs_file="${data_dir}/Ou_Dataset/15_test.json"
dataset_dir="../TSTRL/dataset_dir"
model_dir="./model_dir_mol_parsing"
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python main.py --train_mol_file=$train_mol_file \
                                    --eval_mol_file=$eval_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --test_hu_rs_file=$eval_hu_rs_file \
                                    --test_ou_rs_len5_file=$eval_ou5_rs_file \
                                    --test_ou_rs_len10_file=$eval_ou10_rs_file \
                                    --test_ou_rs_len15_file=$eval_ou15_rs_file \
                                    --dataset_dir=$dataset_dir \
                                    --eval_mol_pool_size 20 \
                                    --TST_model_path "${model_dir}/mol_parsing_bert_ssa" \
                                    --num_layers 1 --max_edu_dist 16 \
