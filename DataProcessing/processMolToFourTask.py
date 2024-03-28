# -*- encoding: utf-8 -*-
'''
file       :processMolToFourTask.py
Description: 将molweni数据集处理成MPC-BERT的类型，从而可以进行多任务
Date       :2024/03/28 14:00:01
Author     :Yaxin Fan
Email      : yxfansuda@stu.suda.edu.cn
'''
"""

将这条数据处理成 下列格式
{
        "edus": [
            {
                "text": "llutz , you understand what z3r0-0n3 wants ? i thought bridging was something else slightly different",
                "speaker": "airtonix"
            },
            {
                "text": "for me it sounds liek bridging , i 'm just not sure about the `` switch connection '' part",
                "speaker": "llutz"
            },
            {
                "text": "i want to be able to switch from ethernet to wireless without interrupting my vpn connection",
                "speaker": "z3r0-0n3"
            },
            {
                "text": "thats the part i 'm not sure about",
                "speaker": "llutz"
            },
            {
                "text": "hmmhow would thermal issues affect this then ?",
                "speaker": "yorick"
            },
            {
                "text": "run memtest to make sure , memory is ok",
                "speaker": "llutz"
            },
            {
                "text": "sys : 30.0c , cpu : 55.5c , aux : 50.5c",
                "speaker": "yorick"
            },
            {
                "text": "this time , it dropped me into the login screen after some garbage was shown on the screen",
                "speaker": "yorick"
            },
            {
                "text": "you 'd better ask in some hardware-related channels",
                "speaker": "llutz"
            }
        ],
        "id": "1056",
        "relations": [
            {
                "y": 1,
                "x": 0,
                "type": "QAP"
            },
            {
                "y": 2,
                "x": 1,
                "type": "Elaboration"
            },
            {
                "y": 3,
                "x": 2,
                "type": "Acknowledgement"
            },
            {
                "y": 4,
                "x": 3,
                "type": "Clarification_question"
            },
            {
                "y": 5,
                "x": 4,
                "type": "QAP"
            },
            {
                "y": 6,
                "x": 5,
                "type": "Continuation"
            },
            {
                "y": 7,
                "x": 5,
                "type": "Result"
            },
            {
                "y": 8,
                "x": 7,
                "type": "Comment"
            }
        ]
    },

{"context": ["did that once , had to reinstall..was using synaptic and deleted some important things.would n't boot", "i do n't think it 's possible to un-install things using the package manager and make the system un-usable . probably the worst you can do is make it annoying to use EMOJI", "if you want a more lightweight solution , i suggest not using gnome or kde or similar . use som lightweight wm as xmonad , and add only the tools you need", "or you could use a gui with lower memory usage", "really ? what did you remove ?"], 
"relation_at": [[1, 0], [2, 0], [3, 0], [4, 0]], 
"ctx_spk": [1, 2, 3, 4, 2], 
"ctx_adr": [-1, 1, 1, 1, 1], 
"answer": "try apt-get installing bum , it 'll show the services you have running and lets you turn the ones you do n't use", 
"ans_idx": 0, # 这个好像没有用到
"ans_spk": 5, 
"ans_adr": 1}

"""
import json
from tqdm import tqdm

class ConvMolforMultTask:
    def __init__(self) -> None:
        """
        只找reply-to structures
        """
        pass

    def load_json(self, file):
        with open(file, 'r', encoding='utf8')as fr:
            datas = json.load(fr)
        return datas
    
    def extract_text_speakers(self, example):
        edus = example['edus']
        text_list = []
        speaker_list = []
        for edu in edus:
            text_list.append(edu['text'])
            speaker_list.append(edu['speaker'])
        # 讲speaker 转为 ids
        spk_dic = {}
        index = 1
        for spk in speaker_list:
            if spk not in spk_dic:
                spk_dic[spk] = index
                index+=1
        speaker_list = [spk_dic[a] for a in speaker_list]    
        return text_list, speaker_list

    def extract_relations(self, example, speaker_list=None):
        relations = example['relations']
        relation_list = []
        relation_dic = {}
        for rela in relations:
            #过滤掉非reply-to structures
            if speaker_list and speaker_list[rela['x']] != speaker_list[rela['y']]:
                relation_list.append((rela['x'], rela['y']))
            else:
                relation_list.append((rela['x'], rela['y']))
        sorted_relation_list  = sorted(relation_list,key=lambda x: x[-1])
        for rela in sorted_relation_list:
            relation_dic[rela[-1]] = rela[0]
        return relation_dic

    def write(self, src_file,des_file):
        datas = self.load_json(src_file)
        new_datas = []
        with open(des_file,'w',encoding ='utf8') as fw:
            for da in tqdm(datas):
                temp_dic = {}
                
                text_list, speaker_list = self.extract_text_speakers(da)
                new_datas.append(len(text_list))
                relation_dic = self.extract_relations(da, speaker_list=speaker_list)
                context = text_list[:-1]
                ctx_spk = speaker_list[:-1]
                answer = text_list[-1]
                ans_spk = speaker_list[-1]
                relation_at = [[key, value] for key, value in relation_dic.items() if key !=len(context) and value !=len(context)]# key,value都是索引
                
                ctx_adr = []
                for i in range(len(context)):
                    if i in relation_dic:
                        ctx_adr.append(speaker_list[relation_dic[i]])
                    else:
                        ctx_adr.append(-1)
                if len(context) in relation_dic:
                    ans_adr = speaker_list[relation_dic[len(context)]]
                else:
                    ans_adr = -1
                temp_dic['context'] = context
                temp_dic['relation_at'] = relation_at
                temp_dic['ctx_spk'] = ctx_spk
                temp_dic['ctx_adr'] = ctx_adr
                temp_dic['answer'] = answer
                temp_dic['ans_spk'] = ans_spk
                temp_dic['ans_adr'] = ans_adr
                assert len(context)==len(ctx_spk)==len(ctx_adr)
                assert len(context)>=len(relation_at)
                
                fw.write(json.dumps(temp_dic, ensure_ascii=False)+'\n')
        print(set(new_datas))
if __name__ =='__main__':
    conv = ConvMolforMultTask()
    conv.write(src_file = '../../TSTRL/Data/Molweni/dev.json',
              des_file='../../TSTRL/Data/Molweni/devFor4Task.json')


