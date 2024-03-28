# coding=utf-8
import json
import random
import numpy as np
import collections
from tqdm import tqdm
from transformers import AutoTokenizer


""" Hu et al. GSN: A Graph-Structured Network for Multi-Party Dialogues. IJCAI 2019. """
class hu_Configs:
    def __init__(self) -> None:
        self.train_file = "../../TSTRL/Data/Hu_Dataset/train.json"
        self.valid_file =  "../../TSTRL/Data/Hu_Dataset/dev.json"
        self.test_file = "../../TSTRL/Data/Hu_Dataset/test.json"
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

class ou10_Configs:
    def __init__(self) -> None:
        self.train_file = "../../TSTRL/Data/Ou_Dataset/10_train.json"
        self.valid_file =  "../../TSTRL/Data/Ou_Dataset/10_dev.json"
        self.test_file = "../../TSTRL/Data/Ou_Dataset/10_test.json"
        self.max_seq_length = 260
        self.max_utr_num = 10
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yxfan/pretrained_model/bert-base-uncased")

class ou15_Configs:
    def __init__(self) -> None:
        self.train_file = "../../TSTRL/Data/Ou_Dataset/15_train.json"
        self.valid_file =  "../../TSTRL/Data/Ou_Dataset/15_dev.json"
        self.test_file = "../../TSTRL/Data/Ou_Dataset/15_test.json"
        self.max_seq_length = 380
        self.max_utr_num = 15
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yxfan/pretrained_model/bert-base-uncased")

""" 
Ouchi et al. Addressee and Response Selection for Multi-Party Conversation. EMNLP 2016.
relesed the original dataset which is composed of 3 experimental settings according to conversation lengths.

In our experiments, we used the version processed and used in 
Le et al. Who Is Speaking to Whom? Learning to Identify Utterance Addressee in Multi-Party Conversations. EMNLP 2019. 
"""

# Length-5
# tf.flags.DEFINE_string("train_file", "./data/emnlp2016/5_train.json", 
#                        "path to train file")
# tf.flags.DEFINE_string("valid_file", "./data/emnlp2016/5_dev.json", 
#                        "path to valid file")
# tf.flags.DEFINE_string("test_file", "./data/emnlp2016/5_test.json", 
#                        "path to test file")
# tf.flags.DEFINE_integer("max_seq_length", 120, 
#                         "max sequence length of concatenated context and response")
# tf.flags.DEFINE_integer("max_utr_num", 5, 
#                         "Maximum utterance number.")

# Length-10
# tf.flags.DEFINE_string("train_file", "./data/emnlp2016/10_train.json", 
#                        "path to train file")
# tf.flags.DEFINE_string("valid_file", "./data/emnlp2016/10_dev.json", 
#                        "path to valid file")
# tf.flags.DEFINE_string("test_file", "./data/emnlp2016/10_test.json", 
#                        "path to test file")
# tf.flags.DEFINE_integer("max_seq_length", 220, 
#                         "max sequence length of concatenated context and response")
# tf.flags.DEFINE_integer("max_utr_num", 10, 
#                         "Maximum utterance number.")

# Length-15
# tf.flags.DEFINE_string("train_file", "./data/emnlp2016/15_train.json", 
#                        "path to train file")
# tf.flags.DEFINE_string("valid_file", "./data/emnlp2016/15_dev.json", 
#                        "path to valid file")
# tf.flags.DEFINE_string("test_file", "./data/emnlp2016/15_test.json", 
#                        "path to test file")
# tf.flags.DEFINE_integer("max_seq_length", 320, 
#                         "max sequence length of concatenated context and response")
# tf.flags.DEFINE_integer("max_utr_num", 15, 
#                         "Maximum utterance number.")

# tf.flags.DEFINE_string("vocab_file", "./uncased_L-12_H-768_A-12/vocab.txt", 
#                        "path to vocab file")
# tf.flags.DEFINE_bool("do_lower_case", True,
#                      "whether to lower case the input text")





def load_dataset(fname):
    dataset = []
    not_in = 0
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            ctx = data['context']
            ctx_spk = data['ctx_spk']
            rsp = data['answer']
            rsp_spk = data['ans_spk']
            assert len(ctx) ==len(ctx_spk)
            
            utrs_same_spk_with_rsp_spk = []
            for utr_id, utr_spk in enumerate(ctx_spk):
                if utr_spk == rsp_spk:
                    utrs_same_spk_with_rsp_spk.append(utr_id)

            if len(utrs_same_spk_with_rsp_spk) == 0:
                continue

            label = [0 for _ in range(len(ctx))]
            for utr_id in utrs_same_spk_with_rsp_spk:
                label[utr_id] = 1
            if sum(label)==0:
                not_in += 1
            dataset.append((ctx, ctx_spk, rsp, rsp_spk, label))
            
    print("dataset_size: {}".format(len(dataset)))
    print("not in: {}".format(not_in))
    return dataset


class InputExample(object):
    def __init__(self, guid, ctx, ctx_spk, rsp, rsp_spk, label):
        """Constructs a InputExample."""
        self.guid = guid
        self.ctx = ctx
        self.ctx_spk = ctx_spk
        self.rsp = rsp
        self.rsp_spk = rsp_spk
        self.label = label


def create_examples(lines, set_type):
    """Creates examples for datasets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, str(i))
        ctx = [utr for utr in line[0]]
        ctx_spk = line[1]
        rsp = line[2]
        rsp_spk = line[3]
        label = line[-1]
        examples.append(InputExample(guid=guid, ctx=ctx, ctx_spk=ctx_spk, rsp=rsp, rsp_spk=rsp_spk, label=label))
    return examples


def truncate_seq_pair(ctx_tokens, rsp_tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        utr_lens = [len(utr_tokens) for utr_tokens in ctx_tokens]
        total_length = sum(utr_lens) + len(rsp_tokens)
        if total_length <= max_length:
            break

        # truncate the longest utterance or response
        if sum(utr_lens) > len(rsp_tokens):
            trunc_tokens = ctx_tokens[np.argmax(np.array(utr_lens))]
        else:
            trunc_tokens = rsp_tokens
        assert len(trunc_tokens) >= 1

        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_sents, input_mask, segment_ids, speaker_ids, cls_positions, rsp_position, label_id):
        self.input_sents = input_sents
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.speaker_ids = speaker_ids
        self.cls_positions = cls_positions
        self.rsp_position = rsp_position
        self.label_id = label_id


def convert_examples_to_features(examples, max_seq_length, max_utr_num, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in tqdm(examples, total=len(examples)):

        ctx_tokens = []
        for utr in example.ctx:
            utr_tokens = tokenizer.tokenize(utr)
            ctx_tokens.append(utr_tokens)
        assert len(ctx_tokens) == len(example.ctx_spk)

        rsp_tokens = tokenizer.tokenize(example.rsp)

        # [CLS]s for context, [CLS] for response, [SEP]
        max_num_tokens = max_seq_length - len(ctx_tokens) - 1 - 1
        truncate_seq_pair(ctx_tokens, rsp_tokens, max_num_tokens)


        tokens = []
        segment_ids = []
        speaker_ids = []
        cls_positions = []
        rsp_position = []

        # utterances
        for i in range(len(ctx_tokens)):
            utr_tokens = ctx_tokens[i]
            utr_spk = example.ctx_spk[i]

            cls_positions.append(len(tokens))
            tokens.append("[CLS]")
            segment_ids.append(0)
            speaker_ids.append(utr_spk)

            for token in utr_tokens:
                tokens.append(token)
                segment_ids.append(0)
                speaker_ids.append(utr_spk)

        # response
        rsp_position.append(len(cls_positions))
        cls_positions.append(len(tokens))
        tokens.append("[CLS]")
        segment_ids.append(0)
        # speaker_ids.append(example.rsp_spk)
        speaker_ids.append(0)  # 0 for mask

        for token in rsp_tokens:
            tokens.append(token)
            segment_ids.append(0)
            # speaker_ids.append(example.rsp_spk)
            speaker_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)
        # speaker_ids.append(example.rsp_spk)
        speaker_ids.append(0)

        
        input_sents = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_sents)
        assert len(input_sents) <= max_seq_length
        while len(input_sents) < max_seq_length:
            input_sents.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
        assert len(input_sents) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length

        assert len(cls_positions) <= max_utr_num
        while len(cls_positions) < max_utr_num:
            cls_positions.append(0)
        assert len(cls_positions) == max_utr_num

        label_id = example.label
        assert len(label_id) <= max_utr_num
        while len(label_id) < max_utr_num:
            label_id.append(0)
        assert len(label_id) == max_utr_num

        features.append(
            InputFeatures(
                input_sents=input_sents,
                input_mask=input_mask,
                segment_ids=segment_ids,
                speaker_ids=speaker_ids,
                cls_positions=cls_positions,
                rsp_position=rsp_position,
                label_id=label_id))
        
    return features


if __name__ == "__main__":
    # hu_config = hu_Configs() 
    hu_config = ou15_Configs()
    # hu_config = hu_Configs() 
    # hu_config = hu_Configs() 
    filenames = [ hu_config.test_file]
    # filenames = [hu_config.train_file, hu_config.valid_file, hu_config.test_file]
    # filetypes = ["train", "valid", "test"]
    filetypes = ["test"]
    for (filename, filetype) in zip(filenames, filetypes):
        dataset = load_dataset(filename)
        examples = create_examples(dataset, filetype)
        features = convert_examples_to_features(examples, hu_config.max_seq_length, hu_config.max_utr_num, hu_config.tokenizer)
        