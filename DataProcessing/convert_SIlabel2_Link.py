# -*- encoding: utf-8 -*-
'''
file       :convert_SIlabel2_Link.py
Description: 将si标签转为link的形式，si即为最后一个last utterance在context中找一个相同的speaker
Date       :2024/03/12 22:29:44
Author     :Yaxin Fan
Email      : yxfansuda@stu.suda.edu.cn
'''


import json
from tqdm import tqdm
from transformers import AutoTokenizer

""" Hu et al. GSN: A Graph-Structured Network for Multi-Party Dialogues. IJCAI 2019. """
class Mol_Configs:
    def __init__(self) -> None:
        self.train_file = "../../TSTRL/Data/Molweni/trainFor4Task.json"
        self.valid_file =  "../../TSTRL/Data/Molweni/devFor4Task.json"
        self.test_file = "../../TSTRL/Data/Molweni/testFor4Task.json"
        self.output_dir = '../../TSTRL/Data/Molweni/Mol_SI_Dir'
        self.max_seq_length = 180
        self.max_utr_num = 7
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yxfan/pretrained_model/bert-base-uncased")

class hu_Configs:
    def __init__(self) -> None:
        self.train_file = "../../TSTRL/Data/Hu_Dataset/train.json"
        self.valid_file =  "../../TSTRL/Data/Hu_Dataset/dev.json"
        self.test_file = "../../TSTRL/Data/Hu_Dataset/test.json"
        self.output_dir = '../../TSTRL/Data/Hu_Dataset/Hu_SI_Dir'
        self.max_seq_length = 180
        self.max_utr_num = 7
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yxfan/pretrained_model/bert-base-uncased")


class ou5_Configs:
    def __init__(self) -> None:
        self.train_file = "../../TSTRL/Data/Ou_Dataset/5_train.json"
        self.valid_file =  "../../TSTRL/Data/Ou_Dataset/5_dev.json"
        self.test_file = "../../TSTRL/Data/Ou_Dataset/5_test.json"
        self.max_seq_length = 130
        self.max_utr_num = 5
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yxfan/pretrained_model/bert-base-uncased")
        self.output_dir = '../../TSTRL/Data/Ou_Dataset/Ou5_SI_Dir'

class ou10_Configs:
    def __init__(self) -> None:
        self.train_file = "../../TSTRL/Data/Ou_Dataset/10_train.json"
        self.valid_file =  "../../TSTRL/Data/Ou_Dataset/10_dev.json"
        self.test_file = "../../TSTRL/Data/Ou_Dataset/10_test.json"
        self.max_seq_length = 260
        self.max_utr_num = 10
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yxfan/pretrained_model/bert-base-uncased")
        self.output_dir = '../../TSTRL/Data/Ou_Dataset/Ou10_SI_Dir'

class ou15_Configs:
    def __init__(self) -> None:
        self.train_file = "../../TSTRL/Data/Ou_Dataset/15_train.json"
        self.valid_file =  "../../TSTRL/Data/Ou_Dataset/15_dev.json"
        self.test_file = "../../TSTRL/Data/Ou_Dataset/15_test.json"
        self.max_seq_length = 380
        self.max_utr_num = 15
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yxfan/pretrained_model/bert-base-uncased")
        self.output_dir = '../../TSTRL/Data/Ou_Dataset/Ou15_SI_Dir'

def load_dataset(fname):
    dataset = []
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            ctx = data['context']
            ctx_spk = data['ctx_spk']
            rsp = data['answer']
            rsp_spk = data['ans_spk']
            integrate_ctx = ctx + [rsp]
            integrate_ctx_spk = ctx_spk + [rsp_spk]
            assert len(integrate_ctx) == len(integrate_ctx_spk)
           
            utrs_same_spk_with_rsp_spk = []
            for utr_id, utr_spk in enumerate(ctx_spk):
                if utr_spk == rsp_spk:
                    utrs_same_spk_with_rsp_spk.append(utr_id)

            if len(utrs_same_spk_with_rsp_spk) == 0:
                continue

            label_last = [0 for _ in range(len(integrate_ctx))]
            for utr_id in utrs_same_spk_with_rsp_spk:
                label_last[utr_id] = 1
            label_matrix = []
            for i in range(len(integrate_ctx)):
                label_matrix.append([0]*len(integrate_ctx))
            label_matrix.pop(-1)
            label_matrix.append(label_last)
            
            dataset.append((ctx, ctx_spk, rsp, rsp_spk, label_matrix))
            # print(data)
            # for label in label_matrix:
            #     print(label)
    print("dataset_size: {}".format(len(dataset)))
    return dataset

def create_examples(lines, set_type, data_type):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s-%s" % (data_type, set_type, str(i))
        ctx =line[0]
        ctx_spk = line[1]
        rsp = line[2]
        rsp_spk = line[3]
        label = line[-1]
        examples.append([guid, ctx, ctx_spk, rsp, rsp_spk, label])
    return examples

def write_des(data, des_file):
    with open(des_file, 'w', encoding='utf8') as fw:
        for da in tqdm(data):
            fw.write(json.dumps(da,ensure_ascii=False)+'\n')


class Convert_ARlabel2_Link:
    def __init__(self):
        pass

    def load_json(self, file):
        with open(file,'r',encoding='utf8')as fr:
            lines  = fr.readlines()
        datas = [ ]
        for line in lines:
            line = line.strip()
            datas.append(json.loads(line))
        return datas

    def convert_labelMatrix_2_relations(self, LabelMatrix, multi_edge=False):
        relations = []
        for j, templabel in enumerate(LabelMatrix):
            i_list = []
            for i, label in enumerate(templabel):
                if label ==1:
                    i_list.append(i)
            if i_list:
                if multi_edge:
                    for last_i in i_list:
                        relations.append({'type': "Result",
                                          'x': last_i,
                                          'y': j})
                else:
                    last_i = i_list[-1]
                    relations.append({'type':"Result",
                                      'x': last_i,
                                      'y': j})
        return relations

    def write_des(self, source_file, des_file, multi_edge = False):
        """

        """
        source_datas = self.load_json(source_file)
        des_datas = []
        for da in tqdm(source_datas):
            # print(da)
            id = da[0]
            edus = da[1]+[da[3]]
            speakers = da[2]+[da[4]]
            LabelMatrix = da[5]
            relations = self.convert_labelMatrix_2_relations(LabelMatrix, multi_edge = multi_edge)
            edu_speakers = []
            for edu, speaker in zip(edus, speakers):
                temp_dic = {'speaker': str(speaker),
                            'text': edu}
                edu_speakers.append(temp_dic)
            des_datas.append({'id':id,
                              'edus': edu_speakers,
                              'relations': relations})
        with open(des_file,'w',encoding='utf8')as fw:
            json.dump(des_datas, fw, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    mol = Mol_Configs()
    hu = hu_Configs()
    ou5 = ou5_Configs()
    ou10 = ou10_Configs()
    ou15 = ou15_Configs()
    dataset_files = [mol, ou10, ou5, hu, ou15]
    dataset_names = ['mol', "ou10", "ou5", "hu", "ou15"]
    for dataset_name, dataset_file in zip(dataset_names, dataset_files):
    # dataset_name ='ohu'
    # for dataset_file in [hu]:
        filenames = [dataset_file.train_file, dataset_file.valid_file, dataset_file.test_file]
        filetypes = ["train", 'valid', 'test']
        for (filename, filetype) in zip(filenames, filetypes):
            dataset = load_dataset(filename)
            examples = create_examples(dataset, filetype, dataset_name)
            new_filename = dataset_file.output_dir + "/{}_si.json".format(filetype)
            write_des(examples, des_file=new_filename)
            conaddParsingType  = Convert_ARlabel2_Link()
            conaddParsingType.write_des(new_filename,
                                        dataset_file.output_dir + "/{}_si_parsingType.json".format(filetype),
                                        multi_edge=False)
            
        break
            # conaddParsingType.write_des('./Ou_Dataset/Ou10_AR_dir/train_ar.json',
            #                             './Ou_Dataset/Ou10_Link_Dir/train.json')

            # conaddParsingType.write_des('./Ou_Dataset/Ou15_AR_dir/train_ar.json',
            #                             './Ou_Dataset/Ou15_Link_Dir/train.json')