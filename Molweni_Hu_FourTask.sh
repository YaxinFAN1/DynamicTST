#!/usr/bin/env bash
data_dir='../TSTRL/Data'
MaskDir='./FourTask_CatClsNodeMaskPath'
train_mol_file="${data_dir}/Molweni/train_reduce_50.json"
eval_mol_file="${data_dir}/Molweni/dev.json"
test_mol_file="${data_dir}/Molweni/test.json"
train_hu_ar_file="${data_dir}/Hu_Dataset/Hu_Link_Dir/train_reduce_50.json"
train_ou5_ar_file="${data_dir}/Ou_Dataset/Ou5_Link_Dir/train_reduce_50.json"
train_ou10_ar_file="${data_dir}/Ou_Dataset/Ou10_Link_Dir/train_reduce_50.json"
train_ou15_ar_file="${data_dir}/Ou_Dataset/Ou15_Link_Dir/train_reduce_50.json"
eval_hu_ar_file="${data_dir}/Hu_Dataset/Hu_Link_Dir/valid.json"
eval_ou5_ar_file="${data_dir}/Ou_Dataset/Ou5_Link_Dir/valid.json"
eval_ou10_ar_file="${data_dir}/Ou_Dataset/Ou10_Link_Dir/valid.json"
eval_ou15_ar_file="${data_dir}/Ou_Dataset/Ou15_Link_Dir/valid.json"
train_hu_rs_file="${data_dir}/Hu_Dataset/train_reduce_50.json"
train_ou5_rs_file="${data_dir}/Ou_Dataset/5_train_reduce_50.json"
train_ou10_rs_file="${data_dir}/Ou_Dataset/10_train_reduce_50.json"
train_ou15_rs_file="${data_dir}/Ou_Dataset/15_train_reduce_50.json"
eval_hu_rs_file="${data_dir}/Hu_Dataset/dev.json"
eval_ou5_rs_file="${data_dir}/Ou_Dataset/5_dev.json"
eval_ou10_rs_file="${data_dir}/Ou_Dataset/10_dev.json"
eval_ou15_rs_file="${data_dir}/Ou_Dataset/15_dev.json"
train_hu_si_file="${data_dir}/Hu_Dataset/Hu_SI_Dir/train_si_parsingType_reduce_50.json"
train_ou5_si_file="${data_dir}/Ou_Dataset/Ou5_SI_Dir/train_si_parsingType_reduce_50.json"
train_ou10_si_file="${data_dir}/Ou_Dataset/Ou10_SI_Dir/train_si_parsingType_reduce_50.json"
train_ou15_si_file="${data_dir}/Ou_Dataset/Ou15_SI_Dir/train_si_parsingType_reduce_50.json"
eval_hu_si_file="${data_dir}/Hu_Dataset/Hu_SI_Dir/valid_si_parsingType.json"
eval_ou5_si_file="${data_dir}/Ou_Dataset/Ou5_SI_Dir/valid_si_parsingType.json"
eval_ou10_si_file="${data_dir}/Ou_Dataset/Ou10_SI_Dir/valid_si_parsingType.json"
eval_ou15_si_file="${data_dir}/Ou_Dataset/Ou15_SI_Dir/valid_si_parsingType.json"
mol_parsing_mask_path="${MaskDir}/mol_parsing_mask_path.pt"
hu_ar_mask_path="${MaskDir}/hu_ar_mask_path.pt"
hu_si_mask_path="${MaskDir}/hu_si_mask_path.pt"
hu_rs_mask_path="${MaskDir}/hu_rs_mask_path.pt"
dataset_dir="./Mol_Hu_FourTask"
model_dir="./model_dir_Mol_Hu_FourTask"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
CUDA_VISIBLE_DEVICES=${GPU}  nohup python -u main.py \
                                    --train_mol_file=$train_mol_file \
                                    --eval_mol_file=$eval_mol_file \
                                    --train_hu_ar_file=$train_hu_ar_file \
                                    --eval_hu_ar_file=$eval_hu_ar_file \
                                    --train_hu_si_file=$train_hu_si_file \
                                    --eval_hu_si_file=$eval_hu_si_file \
                                    --train_hu_rs_file=$train_hu_rs_file \
                                    --eval_hu_rs_file=$eval_hu_rs_file \
                                    --hu_rs_mask_path=$hu_rs_mask_path \
                                    --hu_si_mask_path=$hu_si_mask_path \
                                    --hu_ar_mask_path=$hu_ar_mask_path \
                                    --mol_parsing_mask_path=$mol_parsing_mask_path \
                                    --hu_pool_size 256 \
                                    --hu_batch_size 100 \
                                    --utt_max_len 24 \
                                    --max_mol_text_len 380 \
                                    --max_hu_text_len 380 \
                                    --dataset_dir=$dataset_dir  \
                                    --do_train \
                                    --ST_epoches 15 \
                                    --TST_epoches 5 \
                                    --TrainingParsingTimes 1 \
                                    --ParsingSeperate \
                                    --only_structurePath_CLS_RS \
                                    --ST_model_path "${model_dir}/Mol_Hu_FourTask_ST_only_structurePath_CLS_RS_TrainingParsingTimes1ParsingSeperateEpoch5" \
                                    --TST_model_path "${model_dir}/Mol_Hu_FourTask_TST_only_structurePath_CLS_RS_TrainingParsingTimes1ParsingSeperateEpoch5" \
                                    --seed 65534 > ./logs/Mol_Hu_FourTask_ST_only_structurePath_CLS_RS_TrainingParsingTimes1ParsingSeperateEpoch5.log 2>&1 &