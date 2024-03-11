#!/usr/bin/env bash
data_dir='../TSTRL-two-step/Data'
train_mol_file="${data_dir}/Molweni/train.json"
eval_mol_file="${data_dir}/Molweni/dev.json"
test_mol_file="${data_dir}/Molweni/test.json"
train_hu_file='${data_dir}/Hu_Dataset/Hu_Link_Dir/selectedData.json'
train_ou5_file='${data_dir}/Ou_Dataset/Ou5_Link_Dir/selectedData.json'
train_ou10_file='${data_dir}/Ou_Dataset/Ou10_Link_Dir/selectedData.json'
train_ou15_file='${data_dir}/Ou_Dataset/Ou15_Link_Dir/selectedData.json'
dataset_dir="../TSTRL-two-step/dataset_dir"
model_dir="./model_dir"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
CUDA_VISIBLE_DEVICES=${GPU}  nohup python  -u main.py \
                                    --train_mol_file=$train_mol_file \
                                    --eval_mol_file=$eval_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --train_hu_file=$train_hu_file \
                                    --train_ou5_file=$train_ou5_file \
                                    --train_ou10_file=$train_ou10_file \
                                    --train_ou15_file=$train_ou15_file \
                                    --dataset_dir=$dataset_dir  \
                                    --do_train \
                                    --debug \
                                    --ST_model_path "${model_dir}/model_ST" \
                                    --TST_model_path "${model_dir}/model_TST" \
                                    --seed 65534 > TST.log 2>&1 &