# -*- encoding: utf-8 -*-
'''
file       :SA_BERT.py
Description:
Date       :2024/03/14 11:14:38
Author     :Yaxin Fan
Email      : yxfansuda@stu.suda.edu.cn
'''


# 预训练版本和非预训练版本
# 预训练则把spk embedding 在之前和token_id 相加
# 非预训练版本则把spk_embedding bertmodel 后面

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class BertWithSpeakerID(nn.Module):
    def __init__(self, bert_model_name, speaker_id_dim, num_speakers):
        super(BertWithSpeakerID, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.speaker_embeddings = nn.Embedding(num_speakers, speaker_id_dim)
        
        # 假设BERT的hidden_size与speaker_id_dim一致
        self.embedding_size = self.bert.config.hidden_size

    def forward(self, input_ids, token_type_ids, attention_mask, speaker_ids):
        # 获取BERT的基础embeddings
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)
        position_embeddings = self.bert.embeddings.position_embeddings(torch.arange(input_ids.size(1), device=input_ids.device).expand((input_ids.size(0), input_ids.size(1))))
        
        # 获取speaker_id embeddings
        speaker_embeds = self.speaker_embeddings(speaker_ids)

        # 将所有embeddings相加
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings + speaker_embeds

        # 继续进行BERT的剩余部分
        outputs = self.bert(inputs_embeds =embeddings, attention_mask=attention_mask)
        return outputs




# class BertWithSpeakerID(nn.Module):
#     def __init__(self, bert_model_name, speaker_id_dim, num_speakers):
#         super(BertWithSpeakerID, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.speaker_embeddings = nn.Embedding(num_speakers, speaker_id_dim)
        
#         # 假设BERT的hidden_size与speaker_id_dim一致
#         self.embedding_size = self.bert.config.hidden_size

#     def forward(self, input_ids, token_type_ids, attention_mask, speaker_ids):
#         # 获取BERT的基础embeddings
#         # inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
#         # token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)
#         # position_embeddings = self.bert.embeddings.position_embeddings(torch.arange(input_ids.size(1), device=input_ids.device).expand((input_ids.size(0), input_ids.size(1))))
        
#         # 获取speaker_id embeddings
#         speaker_embeds = self.speaker_embeddings(speaker_ids)

#         # 将所有embeddings相加
#         # embeddings = inputs_embeds + token_type_embeddings + position_embeddings + speaker_embeds

#         # 继续进行BERT的剩余部分
#         outputs = self.bert(input_ids = input_ids,
#                              attention_mask = attention_mask, 
#                              token_type_ids = token_type_ids)
#         print(outputs)
#         print(speaker_embeds.shape)
#         outputs[0] = outputs[0] + speaker_embeds

#         return outputs


if __name__ =='__main__':
    # 假设参数
    num_speakers = 10  # 假设有10个不同的说话人
    speaker_id_dim = 768  # 使用BERT的hidden_size作为embedding维度

    # 实例化模型
    model = BertWithSpeakerID('/home/yxfan/pretrained_model/bert-base-uncased/', speaker_id_dim, num_speakers)
    