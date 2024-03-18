import re
import json
import torch
import numpy as np
from itertools import product
from operator import itemgetter
from torch.utils.data import Dataset
import random
class DiscourseGraph:
    def __init__(self, dialogue, pairs,mask_last_speaker = False):
        self.dialogue = dialogue
        self.mask_last_speaker = mask_last_speaker
        self.pairs = pairs
        self.edu_num = len(self.dialogue['edus'])
        self.paths = self.get_graph(pairs, self.edu_num)
        self.speaker_paths = self.get_speaker_paths(dialogue, self.mask_last_speaker)
        self.turn_paths = self.get_turn_paths(dialogue)

    @staticmethod
    def print_path(path):
        for row in path:
            print([col for col in row])

    def get_speaker_paths(self, dialogue, mask_last_speaker):
        speaker_size = len(dialogue['edus']) + 1
        speaker_4edu = ['None']
        for edu in dialogue['edus']:
            if isinstance(edu['speaker'], str):
                speaker_4edu.append(edu['speaker'])
            else:
                speaker_4edu.append('None')
        # if  self.mask_last_speaker:
        #     # print('before mask')
        #     # print(speaker_4edu)
        #     continue
        if mask_last_speaker:
            speaker_4edu[-1] = 'None'

        # if  self.mask_last_speaker:
        #     print('after mask')
        #     print(speaker_4edu)
        speaker_4edu = np.array(speaker_4edu)
        speaker_4edu_Aside = speaker_4edu.repeat(speaker_size).reshape(speaker_size, speaker_size)
        speaker_4edu_Bside = speaker_4edu_Aside.transpose()
        return (speaker_4edu_Aside == speaker_4edu_Bside).astype(np.long)

    @staticmethod
    def get_turn_paths(dialogue):
        turn_size = len(dialogue['edus']) + 1
        turns = [0] + [edu['turn'] for edu in dialogue['edus']]
        turns = np.array(turns)
        turn_Aside = turns.repeat(turn_size).reshape(turn_size, turn_size)
        turn_Bside = turn_Aside.transpose()
        return (turn_Aside == turn_Bside).astype(np.long)

    @staticmethod
    def get_coreference_path(dialogue):
        coreferences = []
        edu_num = len(dialogue['edus'])
        path = np.zeros((edu_num + 1, edu_num + 1), dtype=np.long)
        if 'solu' in dialogue:
            for cluster in dialogue['solu']:
                coreferences.append([k for (k, v) in cluster])
            for cluster in coreferences:
                for (x, y) in list(product(cluster, cluster)):
                    if x != y:
                        x, y = x + 1, y + 1
                        path[x][y] = 1
        return path.tolist()

    @staticmethod
    def get_graph(pairs, edu_num):
        node_num = edu_num + 1
        graph = np.zeros([node_num, node_num], dtype=np.long)
        for (x, y), label in pairs.items():
            graph[y + 1][x + 1] = label
        return graph.tolist()


class DialogueDataset(Dataset):
    def __init__(self, args,filename, mode, tokenizer,text_max_sep_len, total_seq_len, mask_last_speaker = False):
        print(filename)
        self.mask_last_speaker = mask_last_speaker# 来遮住最后一个speaker，防止信息泄露
        with open(filename, 'r') as file:
            print('loading {} data from {}'.format(mode, filename))
            dialogues = json.load(file)
        print('dialogue numbers')
        print(len(dialogues))
        self.total_seq_len = total_seq_len
        self.text_max_sep_len = text_max_sep_len
        self.tokenizer = tokenizer
        self.padding_value = tokenizer.pad_token_id
        self.dialogues, self.relations = self.format_dialogue(dialogues)
        self.type2ids, self.id2types = None, None


    def __truncate(self, tokens_a, max_seq_len=64):
        while len(tokens_a) > max_seq_len-1:
            tokens_a.pop()

    def format_dialogue(self, dialogues):
        print('format dataset..')
        relation_types = set()
        for dialogue in dialogues:
            last_speaker = None
            turn = 0
            for edu in dialogue['edus']:
                text = edu['text']
                while text.find("http") >= 0:
                    i = text.find("http")
                    j = i
                    while j < len(text) and text[j] != ' ': j += 1
                    text = text[:i] + " [url] " + text[j + 1:]
                invalid_chars = ["/", "\*", "^", ">", "<", "\$", "\|", "=", "@"]
                for ch in invalid_chars:
                    text = re.sub(ch, "", text)
                edu['text'] = text
                if edu["speaker"] != last_speaker:
                    last_speaker = edu["speaker"]
                    turn += 1
                edu["turn"] = turn
            dialogue['relations'] = sorted(dialogue['relations'], key=itemgetter('y', 'x'))
            for relation in dialogue['relations']:
                relation['type'] = relation['type'].strip().lower()
                if relation['type'] not in relation_types:
                    relation_types.add(relation['type'])

        return dialogues, relation_types

    @staticmethod
    def format_relations(relations: set):
        id2types = ['None'] + sorted(list(relations))
        type2ids = {type: i for i, type in enumerate(id2types)}
        return type2ids, id2types

    def get_relations(self, relations, type2ids, id2types):
        self.relations, self.type2ids, self.id2types = relations, type2ids, id2types

    def get_discourse_graph(self):
        for dialogue in self.dialogues:
            pairs = {(relation['x'], relation['y']): self.type2ids[relation['type']]
                     for relation in dialogue['relations']}
            discourse_graph = DiscourseGraph(dialogue, pairs,self.mask_last_speaker)
            dialogue['graph'] = discourse_graph

    @staticmethod
    def nest_padding(sequence):
        max_cols = max([len(row) for batch in sequence for row in batch])
        max_rows = max([len(batch) for batch in sequence])
        sequence = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in sequence]
        sequence = torch.tensor([row + [0] * (max_cols - len(row)) for batch in sequence for row in batch])
        return sequence.reshape(-1, max_rows, max_cols)

    @staticmethod
    def padding(sequence: torch.Tensor, padding_value):
        return (sequence != padding_value).byte()

    def __len__(self):
        return len(self.dialogues)

    def convert_strSpeaker2id(self,speaker_list):
        spkDic = {}
        index = 0
        for spk in speaker_list:
            if spk not in spkDic:
                spkDic[spk] = index
                index += 1
        speaker_ids = []
        for spk in speaker_list:
            speaker_ids.append(spkDic[spk])
        return speaker_ids

    def __getitem__(self, index):
        dialogue = self.dialogues[index]
        texts = [edu['text'] for edu in dialogue['edus']]
        speakers = [edu['speaker'] for edu in dialogue['edus']]
        speakers_ids = self.convert_strSpeaker2id(speakers)
        new_texts  = []
        new_speaker_ids = []
        for text, speaker in zip(texts, speakers_ids):
            text_tokens = self.tokenizer.tokenize(text)
            self.__truncate(text_tokens, max_seq_len=self.text_max_sep_len)
            text_tokens = ['[CLS]'] + text_tokens
            speaker_ids = [speaker]*len(text_tokens)
            new_texts.append(text_tokens)
            new_speaker_ids.append(speaker_ids)
        total_tokens  = []
        total_speaker_ids = []
        for speaker_ids, item in zip(new_speaker_ids, new_texts):
            total_speaker_ids.extend(speaker_ids)
            total_tokens.extend(item)
        total_tokens.append('[SEP]')
        total_speaker_ids.append(new_speaker_ids[-1][-1])
        segment_ids = [0]*len(total_tokens)
        input_mask = [1] * len(total_tokens)
        # print(len(total_tokens))
        gap = self.total_seq_len - len(total_tokens)
        assert gap >0
       
        # fill the gap
        total_tokens = total_tokens + ['[PAD]'] * gap
        segment_ids = segment_ids + [0] * gap
        input_mask = input_mask + [0] * gap
        total_speaker_ids = total_speaker_ids + [0]*gap 
        # print('len_texts')
        # print(len(texts))
        # print(texts)
        # print(total_tokens)
        # print(segment_ids)
        # print(input_mask)
        # print(total_speaker_ids)
        assert len(total_tokens) == self.total_seq_len
        assert len(segment_ids) == self.total_seq_len
        assert len(input_mask) == self.total_seq_len
        assert len(total_speaker_ids) == self.total_seq_len
        temp_sep_index_list = []
        for index, token in enumerate(total_tokens):
            if token == '[CLS]':
                temp_sep_index_list.append(index)
        total_tokens = self.tokenizer.convert_tokens_to_ids(total_tokens)
        total_tokens = torch.LongTensor(total_tokens)
        segment_ids = torch.LongTensor(segment_ids)
        input_mask = torch.FloatTensor(input_mask)
        total_speaker_ids = torch.LongTensor(total_speaker_ids)
        graph = dialogue['graph']
        paths = graph.paths
        pairs = graph.pairs
        speakers = graph.speaker_paths.tolist()
        turns = graph.turn_paths.tolist()
        if 'id' not in dialogue:
            dialogue['id'] = 'none'
        return total_tokens, input_mask, segment_ids, total_speaker_ids, temp_sep_index_list, pairs, paths, speakers, turns, graph.edu_num, dialogue['id']



class RSdataset(Dataset):
    def __init__(self, filename, tokenizer, total_seq_len, text_max_sep_len, filetype='train'):
        self.filetype = filetype
        # print(filename)
        if self.filetype == 'train':
            self.dataset = self.load_dataset(filename, 1)
        elif self.filetype == 'test' or self.filetype == 'eval' :
            self.dataset = self.load_dataset(filename, 9)
        else:
            raise NameError
        self.total_seq_len = total_seq_len 
        self.tokenizer = tokenizer
        self.text_max_sep_len = text_max_sep_len
        self.label_list = ["unfollow", "follow"]


    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break

            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
            else:
                trunc_tokens = tokens_b

            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()


    def load_dataset(self, fname, n_negative):
        ctx_list = []
        ctx_spk_list = []
        rsp_list = []
        rsp_spk_list = []
        with open(fname, 'r') as f:
            for line in f:
                data = json.loads(line)
                ctx_list.append(data['context'])
                ctx_spk_list.append(data['ctx_spk'])
                rsp_list.append(data['answer'])
                rsp_spk_list.append(data['ans_spk'])
        print("matched context-response pairs: {}".format(len(ctx_list)))

        dataset = []
        index_list = list(range(len(ctx_list)))
        for i in range(len(ctx_list)):
            ctx = ctx_list[i]
            ctx_spk = ctx_spk_list[i]
           
            # positive
            rsp = rsp_list[i]
            rsp_spk = rsp_spk_list[i]
            dataset.append((i, ctx, ctx_spk, i, rsp, rsp_spk, 'follow'))

            # negative
            negatives = random.sample(index_list, n_negative)
            while i in negatives:
                negatives = random.sample(index_list, n_negative)
            assert i not in negatives
            neg_spk = str(max([int(a) for a in ctx_spk])+1)# 找一个与众不同的spk
            for n_id in negatives:
                dataset.append((i, ctx, ctx_spk, n_id, rsp_list[n_id], neg_spk, 'unfollow'))

        print("dataset_size: {}".format(len(dataset)))
        return dataset
    
    def __len__(self):
        return len(self.dataset)


    def get_speaker_paths(self, ctx_res, ctx_res_spk):
        speaker_size = len(ctx_res) + 1
        speaker_4edu = ['None']
        for spk in ctx_res_spk:
            if isinstance(spk, str):
                speaker_4edu.append(spk)
            else:
                speaker_4edu.append('None')
        speaker_4edu = np.array(speaker_4edu)
        speaker_4edu_Aside = speaker_4edu.repeat(speaker_size).reshape(speaker_size, speaker_size)
        speaker_4edu_Bside = speaker_4edu_Aside.transpose()
        return (speaker_4edu_Aside == speaker_4edu_Bside).astype(np.long)

    def get_turn_paths(self, ctx_res, ctx_res_spk):
        turn_size = len(ctx_res) + 1
        turn_list = self.get_turn(ctx_res, ctx_res_spk)
        turns = [0] + turn_list
        turns = np.array(turns)
        turn_Aside = turns.repeat(turn_size).reshape(turn_size, turn_size)
        turn_Bside = turn_Aside.transpose()
        return (turn_Aside == turn_Bside).astype(np.long)


    def get_turn(self, ctx_res, ctx_res_spk):
        # print('get turn..')
        last_speaker = None
        turn = 0
        turn_list = []
        for edu, spk in zip(ctx_res, ctx_res_spk):
            if spk != last_speaker:
                last_speaker = spk
                turn += 1
            turn_list.append(turn)
        return turn_list
    
    def __truncate(self, tokens_a, max_seq_len=64):
        while len(tokens_a) > max_seq_len-1:
            tokens_a.pop()

    def __getitem__(self, index):
        # 统一输入和其他任务
        label_map = {}
        for (i, label) in enumerate(self.label_list):  # ['0', '1']
            label_map[label] = i
        example = self.dataset[index]
        # (i, ctx, ctx_spk, i, rsp, rsp_spk, 'follow') 正例
        # (i, ctx, ctx_spk, n_id, rsp_list[n_id], rsp_spk, 'unfollow') 负例
        # print(example)
        texts = example[1] + [example[4]] # context + response
        speakers = example[2] + [example[5]] # ctx_spk + res_spk
        new_texts  = []
        new_speaker_ids = []
        speakers = [int(a)-1 for a in speakers]
        for text, speaker in zip(texts, speakers):
            text_tokens = self.tokenizer.tokenize(text)
            self.__truncate(text_tokens, max_seq_len=self.text_max_sep_len)
            text_tokens = ['[CLS]'] + text_tokens
            speaker_ids = [speaker]*len(text_tokens)
            new_texts.append(text_tokens)
            new_speaker_ids.append(speaker_ids)
        total_tokens  = []
        total_speaker_ids = []
        for speaker_ids, item in zip(new_speaker_ids ,new_texts):
            total_speaker_ids.extend(speaker_ids)
            total_tokens.extend(item)
        total_tokens.append('[SEP]')
        total_speaker_ids.append(new_speaker_ids[-1][-1])
        segment_ids = [0]*len(total_tokens)
        input_mask = [1] * len(total_tokens)
        gap = self.total_seq_len - len(total_tokens)
        # fill the gap
        total_tokens = total_tokens + ['[PAD]'] * gap
        segment_ids = segment_ids + [0] * gap
        input_mask = input_mask + [0] * gap
        total_speaker_ids = total_speaker_ids + [0]*gap 
        assert len(total_tokens) == self.total_seq_len
        assert len(segment_ids) == self.total_seq_len
        assert len(input_mask) == self.total_seq_len
        assert len(total_speaker_ids) == self.total_seq_len
        temp_sep_index_list = []
        for index, token in enumerate(total_tokens):
            if token == '[CLS]':
                temp_sep_index_list.append(index)
        total_tokens = self.tokenizer.convert_tokens_to_ids(total_tokens)
        total_tokens = torch.LongTensor(total_tokens)
        segment_ids = torch.LongTensor(segment_ids)
        input_mask = torch.FloatTensor(input_mask)
        total_speaker_ids = torch.LongTensor(total_speaker_ids)
        paths = ''
        pairs = ''
        speakers = self.get_speaker_paths(texts, speakers).tolist()
        turns = self.get_turn_paths(texts, speakers).tolist()
        ex_id = self.filetype + '_ctxid_{}_resid_{}'.format(example[0], example[3])
        label_id = label_map[example[-1]]
        return total_tokens, input_mask, segment_ids, total_speaker_ids, temp_sep_index_list, pairs, label_id, speakers, turns, len(texts), ex_id

