#!/usr/bin/env bash
data_dir='../TSTRL/Data'
train_mol_file="${data_dir}/Molweni/train_reduce_50.json"
eval_mol_file="${data_dir}/Molweni/dev.json"
test_mol_file="${data_dir}/Molweni/test.json"
test_hu_ar_file="${data_dir}/Hu_Dataset/Hu_Link_Dir/test.json"
test_hu_rs_file="${data_dir}/Hu_Dataset/test.json"
test_hu_si_file="${data_dir}/Hu_Dataset/Hu_SI_Dir/test_si_parsingType.json"


dataset_dir="./Mol_Hu_FourTask"
model_dir="./model_dir_Mol_Hu_FourTask"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
CUDA_VISIBLE_DEVICES=${GPU}  python main.py \
                                    --train_mol_file=$train_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --test_hu_ar_file=$test_hu_ar_file \
                                    --test_hu_rs_file=$test_hu_rs_file \
                                    --train_hu_si_file=$train_hu_si_file \
                                    --test_hu_si_file=$test_hu_si_file \
                                    --hu_pool_size 128 \
                                    --hu_batch_size 100 \
                                    --utt_max_len 24 \
                                    --max_mol_text_len 380 \
                                    --max_hu_text_len 380 \
                                    --dataset_dir=$dataset_dir  \
                                    --num_layers 1 \
                                    --ST_model_path "${model_dir}/Mol_hu_FourTask_ST" \
                                    --TST_model_path "${model_dir}/Mol_hu_FourTask_ST" \
                                    --max_edu_dist 16 
                                 
                                  