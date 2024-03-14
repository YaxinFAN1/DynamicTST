# -*- encoding: utf-8 -*-
'''
file       :ReduceData.py
Description: 将所有训练集都同比缩小50倍，便于快速迭代
Date       :2024/03/14 14:08:07
Author     :Yaxin Fan
Email      : yxfansuda@stu.suda.edu.cn
'''
import json

class ReduceData:
    def __init__(self, src_file, des_file, reduce_rate = 50, type ='') -> None:
        self.dialogues = self.read_file(src_file, file_type=type)
        self.des_file = des_file
        self.reduce_rate = reduce_rate
        self.dia_len = len(self.dialogues)

    def read_file(self, src_file, file_type='rs'):
        if file_type == 'rs':
            with open( src_file, 'r') as fr: 
                lines = fr.readlines()
            dialogues = []
            for line in lines:
                dialogues.append(json.loads(line.strip()))
        else:
            with open( src_file, 'r') as fr:
                dialogues = json.load(fr)
        
        return dialogues
    
    def calculate_remain_num(self):

        return self.dia_len//self.reduce_rate
    

    def sample_datas(self):
        import random
        selected_num = self.calculate_remain_num()
        sampled_examples = random.sample(self.dialogues, selected_num)
        print(len(sampled_examples))
        return sampled_examples 
    
    def write_file(self):
        sample_datas = self.sample_datas()
        with open(self.des_file, 'w', encoding='utf8') as fw:
            json.dump(sample_datas, fw, ensure_ascii=False)


if __name__ == '__main__':


    data_dir='../../TSTRL/Data'
    train_mol_file=data_dir + "/Molweni/train.json"
    reduce_train_mol_file=data_dir + "/Molweni/train_reduce_50.json"
    train_hu_file=data_dir + "/Hu_Dataset/Hu_Link_Dir/train.json"
    reduce_train_hu_file = data_dir + "/Hu_Dataset/Hu_Link_Dir/train_reduce_50.json"

    train_ou5_file=data_dir + "/Ou_Dataset/Ou5_Link_Dir/train.json"
    reduce_train_ou5_file=data_dir + "/Ou_Dataset/Ou5_Link_Dir/train_reduce_50.json"

    train_ou10_file=data_dir + "/Ou_Dataset/Ou10_Link_Dir/train.json"
    reduce_train_ou10_file=data_dir + "/Ou_Dataset/Ou10_Link_Dir/train_reduce_50.json"

    train_ou15_file=data_dir + "/Ou_Dataset/Ou15_Link_Dir/train.json"
    reduce_train_ou15_file=data_dir + "/Ou_Dataset/Ou15_Link_Dir/train_reduce_50.json"

    train_hu_rs_file=data_dir + "/Hu_Dataset/train.json"
    reduce_train_hu_rs_file=data_dir + "/Hu_Dataset/train_reduce_50.json"

    train_ou5_rs_file=data_dir + "/Ou_Dataset/5_train.json"
    reduce_train_ou5_rs_file=data_dir + "/Ou_Dataset/5_train_reduce_50.json"
    train_ou10_rs_file=data_dir + "/Ou_Dataset/10_train.json"
    reduce_train_ou10_rs_file=data_dir + "/Ou_Dataset/10_train_reduce_50.json"
    train_ou15_rs_file=data_dir + "/Ou_Dataset/15_train.json"
    reduce_train_ou15_rs_file=data_dir + "/Ou_Dataset/15_train_reduce_50.json"
    
    reduceD = ReduceData(src_file=train_ou15_rs_file,
                            des_file=reduce_train_ou15_rs_file,
                            type='rs')

    reduceD.write_file() 
    
    