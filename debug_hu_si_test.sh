#!/usr/bin/env bash
data_dir='../TSTRL/Data'
test_mol_file="${data_dir}/Molweni/test.json"
test_hu_ar_file="${data_dir}/Hu_Dataset/Hu_Link_Dir/test.json"
test_ou5_ar_file="${data_dir}/Ou_Dataset/Ou5_Link_Dir/test.json"
test_ou10_ar_file="${data_dir}/Ou_Dataset/Ou10_Link_Dir/test.json"
test_ou15_ar_file="${data_dir}/Ou_Dataset/Ou15_Link_Dir/test.json"
test_hu_si_file="${data_dir}/Hu_Dataset/Hu_SI_Dir/test_si_parsingType.json"
test_ou5_si_file="${data_dir}/Ou_Dataset/Ou5_SI_Dir/test_si_parsingType.json"
test_ou10_si_file="${data_dir}/Ou_Dataset/Ou10_SI_Dir/test_si_parsingType.json"
test_ou15_si_file="${data_dir}/Ou_Dataset/Ou15_SI_Dir/test_si_parsingType.json"
test_hu_rs_file="${data_dir}/Hu_Dataset/test.json"
test_ou5_rs_file="${data_dir}/Ou_Dataset/5_test.json"
test_ou10_rs_file="${data_dir}/Ou_Dataset/10_test.json"
test_ou15_rs_file="${data_dir}/Ou_Dataset/15_test.json"
dataset_dir="../TSTRL/dataset_dir"
model_dir="./model_dir_hu_SI"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
CUDA_VISIBLE_DEVICES=${GPU}   python   main.py \
                                    --train_mol_file=$train_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --test_hu_rs_file=$test_hu_rs_file \
                                    --test_ou_rs_len5_file=$test_ou_rs_len5_file \
                                    --test_ou_rs_len10_file=$test_ou_rs_len10_file \
                                    --test_ou_rs_len15_file=$test_ou_rs_len15_file \
                                    --test_hu_si_file=$test_hu_si_file \
                                    --test_ou5_si_file=$test_ou5_si_file \
                                    --test_ou10_si_file=$test_ou10_si_file \
                                    --test_ou15_si_file=$test_ou15_si_file \
                                    --test_hu_ar_file=$test_hu_ar_file \
                                    --test_ou5_ar_file=$test_ou5_ar_file \
                                    --test_ou10_ar_file=$test_ou10_ar_file \
                                    --test_ou15_ar_file=$test_ou15_ar_file \
                                    --hu_pool_size 10 \
                                    --hu_batch_size 10 \
                                    --dataset_dir=$dataset_dir  \
                                    --source_file=$test_hu_si_file \
                                    --ST_model_path "${model_dir}/hu_si_bert_ssa_withoutGRU" \
                                    --TST_model_path "${model_dir}/hu_si_bert_ssa_withoutGRU" \
                                    --num_layers 1 \
                                    --max_edu_dist 16 