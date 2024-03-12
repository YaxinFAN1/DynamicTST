import torch
import json
import numpy as np

torch.autograd.set_detect_anomaly(True)
# https://discuss.pytorch.org/t/nested-list-of-variable-length-to-a-tensor/38699/21
def pad_tensors(tensors):
    """
    Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

    The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
    where `Si` is the maximum value of dimension `i` amongst all tensors.
    """
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.zeros(padded_dim)
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, :size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, :size[0], :size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
        else:
            raise ValueError('Padding is supported for upto 3D tensors at max.')
    return padded_tensor

def ints_to_tensor(ints):
    """
    Converts a nested list of integers to a padded tensor.
    """
    if isinstance(ints, torch.Tensor):
        return ints
    if isinstance(ints, list):
        if isinstance(ints[0], int):
            return torch.LongTensor(ints)
        if isinstance(ints[0], torch.Tensor):
            return pad_tensors(ints)
        if isinstance(ints[0], list):
            return ints_to_tensor([ints_to_tensor(inti) for inti in ints])

def get_mask(node_num, max_edu_dist):
    batch_size, max_num=node_num.size(0), node_num.max()
    mask=torch.arange(max_num).unsqueeze(0).cuda()<node_num.unsqueeze(1)
    mask=mask.unsqueeze(1).expand(batch_size, max_num, max_num)
    mask=mask&mask.transpose(1,2)
    mask = torch.tril(mask, -1)
    if max_num > max_edu_dist:
        mask = torch.triu(mask, max_edu_dist - max_num)
    return mask

def compute_loss(link_scores, label_scores, graphs, mask, p=False, negative=False):
    link_scores[~mask]=-1e9
    label_mask=(graphs!=0)&mask
    link_mask=label_mask.clone()
    link_scores=torch.nn.functional.softmax(link_scores, dim=-1)
    link_loss=-torch.log(link_scores[link_mask])
    vocab_size=label_scores.size(-1)
    label_loss=torch.nn.functional.cross_entropy(label_scores[label_mask].reshape(-1, vocab_size), graphs[label_mask].reshape(-1), reduction='none')
    if negative:
        negative_mask=(graphs==0)&mask
        negative_loss=torch.nn.functional.cross_entropy(label_scores[negative_mask].reshape(-1, vocab_size), graphs[negative_mask].reshape(-1),reduction='mean')
        return link_loss, label_loss, negative_loss
    if p:
        return link_loss, label_loss, torch.nn.functional.softmax(label_scores[label_mask],dim=-1)[torch.arange(label_scores[label_mask].size(0)),graphs[mask]]
    return link_loss, label_loss

def record_eval_result(eval_matrix, predicted_result):
    for k, v in eval_matrix.items():
        if v is None:
            if isinstance(predicted_result[k], dict):
                eval_matrix[k] = [predicted_result[k]]
            else:
                eval_matrix[k] = predicted_result[k]
        elif isinstance(v, list):
            eval_matrix[k] += [predicted_result[k]]
        else:
            eval_matrix[k] = np.append(eval_matrix[k], predicted_result[k])

def tsinghua_F1(eval_matrix):
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    for hypothesis, reference, edu_num in zip(eval_matrix['hypothesis'], eval_matrix['reference'],
                                              eval_matrix['edu_num']):
        cnt = [0] * edu_num
        for r in reference:
            cnt[r[1]] += 1
        for i in range(edu_num):
            if cnt[i] == 0:
                cnt_golden += 1
        cnt_pred += 1
        if cnt[0] == 0:
            cnt_cor_bi += 1
            cnt_cor_multi += 1
        cnt_golden += len(reference)
        cnt_pred += len(hypothesis)
        for pair in hypothesis:
            if pair in reference:
                cnt_cor_bi += 1
                if hypothesis[pair] == reference[pair]:
                    cnt_cor_multi += 1
    prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return f1_bi, f1_multi

def conv_list2Dic(datalist):
    dataDic = {}
    for da in datalist:
        dataDic[da['id']] = da
    return dataDic

def write_selected_data(id_file, src_file, des_file):
    with open(id_file, 'r', encoding='utf8') as fr:
        lines  = fr.readlines()
    ids_list = list(set(a.strip() for a in lines))
    with open(src_file, 'r', encoding='utf8') as fr:
        src_datas = json.load(fr)
    src_dic = conv_list2Dic(src_datas)
    des_datas = []
    for id in ids_list:
        des_datas.append(src_dic[id])
    with open(des_file,'w',encoding = 'utf8') as fw:
        json.dump(des_datas, fw, ensure_ascii=False)
    

class EvaluateAddressTo:
    """
    判断speaker是否正确，而不是link完全一致。
    先把link转为speaker的形式，然后判断speaker是否一致。
    """
    def __init__(self):
        """
        读取输出文件，计算p@1和Acc
        """
        pass
    def readSourceFile(self, file):
        """
        读取source_file，得到speaker信息,
        :param file:
        :return:
        """
        speaker_dic = {}
        with open(file,'r',encoding='utf8') as fr:
            data = json.load(fr)
        for da in data:
            temp_speaker_dic  = {}
            id = da['id']
            for index, edu in enumerate(da['edus']):
                temp_speaker_dic[index] = edu['speaker']
            speaker_dic[id] = temp_speaker_dic
        return speaker_dic

    def readFile(self, file):
        with open(file,'r',encoding='utf8')as fr:
            lines = fr.readlines()
        data = []
        for line in lines:
            line = line.strip()
            data.append(eval(json.loads(line)))
        new_data_dic  = {}
        not_Same =0
        for da in data:
            temp_dic = {}
            # print(da)
            id = da['id']
            temp_dic['hypothesis']= da['hypothesis']
            temp_dic['reference']  = da['reference']
            temp_dic['edu_num'] = da['edu_num']
            new_data_dic[id] = temp_dic
            if len(da['reference'])!=da['edu_num']-1:
                not_Same+=1
        return new_data_dic

    def calP1(self, datadic, speaker_dic):
        """
        计算p@1，
        :return:
        """
        P_total = P_correct = 0

        for id, value  in datadic.items():
            temp_speaker_dic = speaker_dic[id]
            hypo = value['hypothesis']
            ref = value['reference']

            P_total += len(ref)
            for goldlink in ref:
                if goldlink in hypo:
                    P_correct += 1

        P_At_1 = P_correct/P_total
        print('P_total is {}, P_correct is {}'.format(P_total, P_correct))
        print('P@1 is {}'.format(round(P_At_1,6)))

    def calP1_new(self, datadic, speaker_dic):
        """
        计算p@1，
        :return:
        """
        P_total = P_correct = 0
        for id, value in datadic.items():
            temp_speaker_dic = speaker_dic[id]
            hypo = value['hypothesis']
            ref = value['reference']
            hypoLinks = [list(a) for a in hypo]
            hypoLinks = sorted(hypoLinks, key=lambda x: x[1])
            # 得到当前y的address to,根据x转换得到speaker
            hypo_address_to = []  # [utt,address_to_speaker],[utt,address_to_speaker]
            for link in hypoLinks:
                if link[0]>=0:
                    hypo_address_to.append((link[1], temp_speaker_dic[link[0]]))
            refLinks = [list(a) for a in ref]
            refLinks = sorted(refLinks, key=lambda x: x[1])
            ref_address_to = []  # [utt,address_to_speaker],[utt,address_to_speaker]
            for link in refLinks:
                ref_address_to.append((link[1], temp_speaker_dic[link[0]]))

            P_total += len(set(ref_address_to))
            for a in set(ref_address_to):
                if a in set(hypo_address_to):
                    P_correct += 1

        P_At_1 = P_correct / P_total
        print(P_correct)
        print(P_total)
        # print('P_total is {}, P_correct is {}'.format(P_total, P_correct))
        # print('P@1 is {}'.format(round(P_At_1, 6)))
        return P_At_1
    def IsAallInB(self,set1,set2):
        """
        判断set1中的元素是够全部在set2中
        :param set1:
        :param set2:
        :return:
        """
        Flag = True
        for a in set1:
            if a in set2:
                continue
            else:
                Flag = False
                break
        return Flag

    def calSesseionAcc(self, datadic, speakerdic):
        """
        计算session acc（所有的link都正确才算对）
        :param datadic:
        :return:
        """
        Acc_total = Acc_correct = 0
        for id, value  in datadic.items():
            Acc_total += 1
            # print(_)
            temp_speaker_dic = speakerdic[id]
            hypo = value['hypothesis']
            ref = value['reference']
            hypoLinks = [list(a) for a in hypo]
            hypoLinks = sorted(hypoLinks, key=lambda x: x[1])
            # 得到当前y的address to,根据x转换得到speaker
            hypo_address_to = []  # [utt,address_to_speaker],[utt,address_to_speaker]
            for link in hypoLinks:
                if link[0]>=0:

                    hypo_address_to.append((link[1], temp_speaker_dic[link[0]]))
            refLinks = [list(a) for a in ref]
            refLinks = sorted(refLinks, key=lambda x: x[1])
            ref_address_to = []  # [utt,address_to_speaker],[utt,address_to_speaker]
            for link in refLinks:
                ref_address_to.append((link[1], temp_speaker_dic[link[0]]))
            ref_address_to_set = set(ref_address_to)
            hypo_address_to_set = set(hypo_address_to)
            if self.IsAallInB(ref_address_to_set,hypo_address_to_set):
                Acc_correct += 1
        SessionAcc = Acc_correct / Acc_total
        print(Acc_correct)
        print(Acc_total)
        return SessionAcc

    def test(self, inputfile, sourcetestfile):
        """
        This is a test funciton.
        :return:
        """
        speakerdic = self.readSourceFile(sourcetestfile)
        datadic = self.readFile(inputfile)
        self.calP1_new(datadic, speakerdic)
        self.calSesseionAcc(datadic, speakerdic)

    def get_Pat1AndSessAcc(self,datadic, sourcetestfile):
        print('ssss')
        print(sourcetestfile)
        speakerdic = self.readSourceFile(sourcetestfile)
        Pat1 = self.calP1_new(datadic, speakerdic)
        SessionAcc = self.calSesseionAcc(datadic, speakerdic)
        return Pat1, SessionAcc