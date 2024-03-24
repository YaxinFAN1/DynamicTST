import argparse
import random
import os
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from Model import PolicyNetwork
from dialogue_dataset import DialogueDataset, RSdataset
from SA_BERT import BertWithSpeakerID
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def seed_everything(seed=256):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mol_file', type=str)
    parser.add_argument('--eval_mol_file', type=str)
    parser.add_argument('--test_mol_file', type=str)
    parser.add_argument('--train_hu_ar_file', type=str)
    parser.add_argument('--train_ou5_ar_file', type=str)
    parser.add_argument('--train_ou10_ar_file', type=str)
    parser.add_argument('--train_ou15_ar_file', type=str)
    parser.add_argument('--eval_hu_ar_file', type=str)
    parser.add_argument('--eval_ou5_ar_file', type=str)
    parser.add_argument('--eval_ou10_ar_file', type=str)
    parser.add_argument('--eval_ou15_ar_file', type=str)
    parser.add_argument('--test_hu_ar_file', type=str)
    parser.add_argument('--test_ou5_ar_file', type=str)
    parser.add_argument('--test_ou10_ar_file', type=str)
    parser.add_argument('--test_ou15_ar_file', type=str)
    parser.add_argument('--train_hu_si_file', type=str)
    parser.add_argument('--train_ou5_si_file', type=str)
    parser.add_argument('--train_ou10_si_file', type=str)
    parser.add_argument('--train_ou15_si_file', type=str)
    parser.add_argument('--eval_hu_si_file', type=str)
    parser.add_argument('--eval_ou5_si_file', type=str)
    parser.add_argument('--eval_ou10_si_file', type=str)
    parser.add_argument('--eval_ou15_si_file', type=str)
    parser.add_argument('--test_hu_si_file', type=str)
    parser.add_argument('--test_ou5_si_file', type=str)
    parser.add_argument('--test_ou10_si_file', type=str)
    parser.add_argument('--test_ou15_si_file', type=str)
    parser.add_argument('--train_hu_rs_file', type=str)
    parser.add_argument('--train_ou5_rs_file', type=str)
    parser.add_argument('--train_ou10_rs_file', type=str)
    parser.add_argument('--train_ou15_rs_file', type=str)
    parser.add_argument('--test_hu_rs_file', type=str)
    parser.add_argument('--test_ou_rs_len5_file', type=str)
    parser.add_argument('--test_ou_rs_len10_file', type=str)
    parser.add_argument('--test_ou_rs_len15_file', type=str)
    parser.add_argument('--eval_hu_rs_file', type=str)
    parser.add_argument('--eval_ou_rs_len5_file', type=str)
    parser.add_argument('--eval_ou_rs_len10_file', type=str)
    parser.add_argument('--eval_ou_rs_len15_file', type=str)
    parser.add_argument('--hu_ar_mask_path', type=str)
    parser.add_argument('--hu_si_mask_path', type=str)
    parser.add_argument('--hu_rs_mask_path', type=str)
    parser.add_argument('--mol_parsing_mask_path', type=str)
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--model_name_or_path', type=str, default='/home/yxfan/pretrained_model/bert-base-uncased/')
    parser.add_argument('--remake_dataset', action="store_true")
    parser.add_argument('--remake_mask', action="store_true")
    parser.add_argument('--remake_tokenizer', action="store_true")
    parser.add_argument('--max_edu_dist', type=int, default=20)
    parser.add_argument('--path_hidden_size', type=int, default=384)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_speakers', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--speaker', action='store_true')
    parser.add_argument('--valid_dist', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--pretrained_model_learning_rate', type=float, default=1e-5)
    parser.add_argument('--ST_epoches', type=int, default=5)
    parser.add_argument('--TST_epoches', type=int, default=3)
    parser.add_argument('--RL_epoches', type=int, default=3)
    parser.add_argument('--TrainingParsingTimes', type=int, default=2)
    parser.add_argument('--mol_pool_size', type=int, default=100)
    parser.add_argument('--hu_pool_size', type=int, default=100)
    parser.add_argument('--ou5_pool_size', type=int, default=100)
    parser.add_argument('--ou10_pool_size', type=int, default=100)
    parser.add_argument('--ou15_pool_size', type=int, default=100)
    parser.add_argument('--eval_len5_pool_size', type=int, default=1)
    parser.add_argument('--eval_len10_pool_size', type=int, default=1)
    parser.add_argument('--eval_len15_pool_size', type=int, default=1)
    parser.add_argument('--eval_mol_pool_size', type=int, default=10)
    parser.add_argument('--mol_batch_size', type=int, default=100)
    parser.add_argument('--hu_batch_size', type=int, default=100)
    parser.add_argument('--ou5_batch_size', type=int, default=100)
    parser.add_argument('--ou10_batch_size', type=int, default=100)
    parser.add_argument('--ou15_batch_size', type=int, default=100)
    parser.add_argument('--hu_batch_size_rl', type=int, default=10000)
    parser.add_argument('--ou5_batch_size_rl', type=int, default=10000)
    parser.add_argument('--ou10_batch_size_rl', type=int, default=5000)
    parser.add_argument('--ou15_batch_size_rl', type=int, default=4000)
    parser.add_argument('--ST_model_path', type=str, default='model.pt')
    parser.add_argument('--TST_model_path', type=str, default='model.pt')
    parser.add_argument('--hu_select_id_file', type=str, default='hu_select_id_file.txt')
    parser.add_argument('--ou5_select_id_file', type=str, default='ou5_select_id_file.txt')
    parser.add_argument('--ou10_select_id_file', type=str, default='ou10_select_id_file.txt')
    parser.add_argument('--ou15_select_id_file', type=str, default='ou15_select_id_file.txt')
    parser.add_argument('--hu_selected_data_file', type=str, default='hu_selected_data.json')
    parser.add_argument('--ou5_selected_data_file', type=str, default='ou5_selected_data.json')
    parser.add_argument('--ou10_selected_data_file', type=str, default='ou10_selected_data.json')
    parser.add_argument('--ou15_selected_data_file', type=str, default='ou15_selected_data.json')
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--report_step', type=int, default= 20)
    parser.add_argument('--early_stop', type=int, default=1000)
    parser.add_argument('--utt_max_len', type=int, default= 24)
    parser.add_argument('--max_mol_text_len', type=int, default= 380)
    parser.add_argument('--max_hu_text_len', type=int, default= 180)
    parser.add_argument('--max_ou5_text_len', type=int, default= 130)
    parser.add_argument('--max_ou10_text_len', type=int, default= 260)
    parser.add_argument('--max_ou15_text_len', type=int, default= 380)
    parser.add_argument('--TST_Learning_Mode',  action="store_true")
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--state_dim', type=int, default= 768)
    parser.add_argument('--hdim', type=int, default= 384)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--seed', type=int, default= 512)
    parser.add_argument('--withSpkembedding', action="store_true")
    parser.add_argument('--only_SABERT', action="store_true")
    parser.add_argument('--only_BERT', action="store_true")
    parser.add_argument('--cat_cls_structurePath', action="store_true")
    parser.add_argument('--with_GRU', action="store_true")
    parser.add_argument('--source_file', type=str, default='')
    
    args = parser.parse_args()
    seed_everything(args.seed)
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda")

    if not os.path.isdir(args.dataset_dir):
        os.mkdir(args.dataset_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_hu_ar_file = os.path.join(args.dataset_dir, 'train_hu_ar.pt')
    train_ou5_ar_file = os.path.join(args.dataset_dir, 'train_ou5_ar.pt')
    train_ou10_ar_file = os.path.join(args.dataset_dir, 'train_ou10_ar.pt')
    train_ou15_ar_file = os.path.join(args.dataset_dir, 'train_ou15_ar.pt')
    
    eval_hu_ar_file = os.path.join(args.dataset_dir, 'eval_hu_ar.pt')
    eval_ou5_ar_file = os.path.join(args.dataset_dir, 'eval_ou5_ar.pt')
    eval_ou10_ar_file = os.path.join(args.dataset_dir, 'eval_ou10_ar.pt')
    eval_ou15_ar_file = os.path.join(args.dataset_dir, 'eval_ou15_ar.pt')
    

    train_hu_si_file = os.path.join(args.dataset_dir, 'train_hu_si.pt')
    train_ou5_si_file = os.path.join(args.dataset_dir, 'train_ou5_si.pt')
    train_ou10_si_file = os.path.join(args.dataset_dir, 'train_ou10_si.pt')
    train_ou15_si_file = os.path.join(args.dataset_dir, 'train_ou15_si.pt')
    
    eval_hu_si_file = os.path.join(args.dataset_dir, 'eval_hu_si.pt')
    eval_ou5_si_file = os.path.join(args.dataset_dir, 'eval_ou5_si.pt')
    eval_ou10_si_file = os.path.join(args.dataset_dir, 'eval_ou10_si.pt')
    eval_ou15_si_file = os.path.join(args.dataset_dir, 'eval_ou15_si.pt')

    train_hu_rs_file = os.path.join(args.dataset_dir, 'train_hu_rs.pt')
    train_ou5_rs_file = os.path.join(args.dataset_dir, 'train_ou5_rs.pt')
    train_ou10_rs_file = os.path.join(args.dataset_dir, 'train_ou10_rs.pt')
    train_ou15_rs_file = os.path.join(args.dataset_dir, 'train_ou15_rs.pt')

    eval_hu_rs_file = os.path.join(args.dataset_dir, 'eval_hu_rs.pt')
    eval_ou5_rs_file = os.path.join(args.dataset_dir, 'eval_ou5_rs.pt')
    eval_ou10_rs_file = os.path.join(args.dataset_dir, 'eval_ou10_rs.pt')
    eval_ou15_rs_file = os.path.join(args.dataset_dir, 'eval_ou15_rs.pt')

    train_mol_file = os.path.join(args.dataset_dir, 'train_mol.pt')
    test_mol_file = os.path.join(args.dataset_dir, 'test_mol.pt')
    eval_mol_file = os.path.join(args.dataset_dir, 'eval_mol.pt')

    if os.path.exists(train_mol_file) and not args.remake_dataset:
        print('loading dataset..')
        if args.do_train:
            train_dataset_hu_ar = torch.load(train_hu_ar_file)
            eval_dataset_hu_ar = torch.load(eval_hu_ar_file)

            train_dataset_hu_si = torch.load(train_hu_si_file)
            eval_dataset_hu_si = torch.load(eval_hu_si_file)


            train_dataset_hu_rs = torch.load(train_hu_rs_file)
            eval_dataset_hu_rs = torch.load(eval_hu_rs_file)
            # train_dataset_ou5 = torch.load(train_ou5_file)
            # train_dataset_ou10 = torch.load(train_ou10_file)
            # train_dataset_ou15 = torch.load(train_ou15_file)
            # train_hu_rs_dataset = torch.load(train_hu_rs_file)
            # eval_hu_rs_dataset = torch.load(eval_hu_rs_file)
            # train_ou5_rs_file = torch.load(train_ou5_rs_file)
            # train_ou10_rs_file = torch.load(train_ou10_rs_file)
            # train_ou15_rs_file = torch.load(train_ou15_rs_file)
            eval_dataset_mol = torch.load(eval_mol_file)
            
        train_dataset_mol = torch.load(train_mol_file)
        relations, type2ids, id2types = train_dataset_mol.relations, train_dataset_mol.type2ids, train_dataset_mol.id2types
        if not args.do_train:

            test_dataset_mol = DialogueDataset(args=args, filename=args.test_mol_file, tokenizer=tokenizer,
                                                   mode='test', text_max_sep_len=args.utt_max_len,
                                                   total_seq_len=args.max_mol_text_len)
            
            test_dataset_mol.get_relations(relations, type2ids, id2types)
            test_dataset_mol.get_discourse_graph()

            test_dataset_hu_ar = DialogueDataset(args=args, filename=args.test_hu_ar_file, tokenizer=tokenizer,
                                                    mode='test', text_max_sep_len=args.utt_max_len,
                                                    total_seq_len=args.max_hu_text_len)
            test_dataset_hu_ar.get_relations(relations, type2ids, id2types)
            test_dataset_hu_ar.get_discourse_graph()

            test_dataset_hu_si = DialogueDataset(args=args, filename=args.test_hu_si_file, tokenizer=tokenizer,
                                                    mode='test', text_max_sep_len=args.utt_max_len,
                                                    total_seq_len=args.max_hu_text_len, mask_last_speaker=True)
            test_dataset_hu_si.get_relations(relations, type2ids, id2types)
            test_dataset_hu_si.get_discourse_graph()
               
            test_dataset_hu_rs = RSdataset( 
                                            filename=args.test_hu_rs_file, 
                                            tokenizer=tokenizer,
                                            total_seq_len=args.max_hu_text_len,
                                            text_max_sep_len=args.utt_max_len,
                                            filetype='test')
            # test_dataset_ou_len5 = DialogueDataset(args=args, filename=args.test_ou_len5_file, tokenizer=tokenizer,
            #                                   mode='test', text_max_sep_len=args.utt_max_len,
            #                                   total_seq_len=args.max_ou5_text_len)
           
            # test_dataset_ou_len5.get_relations(relations, type2ids, id2types)
            # test_dataset_ou_len5.get_discourse_graph()
            # RSdataset
            #filename, tokenizer, total_seq_len, text_max_sep_len, filetype='train'
            # test_dataset_ou_rs_len5 = RSdataset(
            #                                 filename=args.test_ou_rs_len5_file, 
            #                                 tokenizer=tokenizer,
            #                                 total_seq_len=args.max_ou5_text_len,
            #                                 text_max_sep_len=args.utt_max_len,
            #                                 filetype='test')

            # test_dataset_ou_rs_len10 = RSdataset(
            #                                 filename=args.test_ou_rs_len10_file, 
            #                                 tokenizer=tokenizer,
            #                                 total_seq_len=args.max_ou10_text_len,
            #                                 text_max_sep_len=args.utt_max_len,
            #                                 filetype='test')
            # test_dataset_ou_rs_len15 = RSdataset(
            #                                 filename=args.test_ou_rs_len15_file, 
            #                                 tokenizer=tokenizer,
            #                                 total_seq_len=args.max_ou15_text_len,
            #                                 text_max_sep_len=args.utt_max_len,
            #                                 filetype='test')
         

    else:
        
        train_dataset_mol = DialogueDataset(args=args, filename=args.train_mol_file, tokenizer=tokenizer, mode='train',
                                        text_max_sep_len=args.utt_max_len,
                                        total_seq_len=args.max_mol_text_len)

        eval_dataset_mol = DialogueDataset(args=args, filename=args.eval_mol_file, tokenizer=tokenizer, mode='eval',
                                           text_max_sep_len=args.utt_max_len,
                                           total_seq_len = args.max_mol_text_len)

        train_dataset_hu_ar = DialogueDataset(args=args, filename= args.train_hu_ar_file, tokenizer=tokenizer, mode='train',text_max_sep_len=args.utt_max_len,
                                           total_seq_len = args.max_mol_text_len)
        
        eval_dataset_hu_ar = DialogueDataset(args=args, filename= args.eval_hu_ar_file, tokenizer=tokenizer, mode='eval',text_max_sep_len=args.utt_max_len,
                                           total_seq_len = args.max_mol_text_len)
        

        train_dataset_hu_si = DialogueDataset(args=args, filename= args.train_hu_si_file, tokenizer=tokenizer, mode='train',text_max_sep_len=args.utt_max_len,
                                           total_seq_len = args.max_mol_text_len,mask_last_speaker=True)
        
        eval_dataset_hu_si = DialogueDataset(args=args, filename= args.eval_hu_si_file, tokenizer=tokenizer, mode='eval',text_max_sep_len=args.utt_max_len,
                                           total_seq_len = args.max_mol_text_len,mask_last_speaker=True)



        train_dataset_hu_rs = RSdataset(
                                            filename=args.train_hu_rs_file, 
                                            tokenizer=tokenizer,
                                            total_seq_len=args.max_mol_text_len,
                                            text_max_sep_len=args.utt_max_len,
                                            filetype='train')
        
        eval_dataset_hu_rs = RSdataset(
                                            filename=args.eval_hu_rs_file, 
                                            tokenizer=tokenizer,
                                            total_seq_len=args.max_mol_text_len,
                                            text_max_sep_len=args.utt_max_len,
                                            filetype='eval')

        # train_dataset_ou5 = DialogueDataset(args=args, filename=args.train_ou5_file, tokenizer=tokenizer,
        #                                          mode='train', text_max_sep_len=args.utt_max_len,
        #                                          total_seq_len=args.max_ou5_text_len)

        # train_dataset_ou10 = DialogueDataset(args=args, filename=args.train_ou10_file, tokenizer=tokenizer,
        #                                     mode='train', text_max_sep_len=args.utt_max_len,
        #                                     total_seq_len=args.max_ou10_text_len)

        # train_dataset_ou15 = DialogueDataset(args=args, filename=args.train_ou15_file, tokenizer=tokenizer,
        #                                     mode='train', text_max_sep_len=args.utt_max_len,
        #                                     total_seq_len=args.max_ou15_text_len)

                    # RSdataset
        #filename, tokenizer, total_seq_len, text_max_sep_len, filetype='train'
        # train_dataset_ou_rs_len5 = RSdataset(
        #                                     filename=args.train_ou_rs_len5_file, 
        #                                     tokenizer=tokenizer,
        #                                     total_seq_len=args.max_ou5_text_len,
        #                                     text_max_sep_len=args.utt_max_len,
        #                                     filetype='train')

        # train_dataset_ou_rs_len10 = RSdataset(
        #                                     filename=args.train_ou_rs_len10_file, 
        #                                     tokenizer=tokenizer,
        #                                     total_seq_len=args.max_ou10_text_len,
        #                                     text_max_sep_len=args.utt_max_len,
        #                                     filetype='train')
        
        # train_dataset_ou_rs_len15 = RSdataset(
        #                                     filename=args.train_ou_rs_len15_file, 
        #                                     tokenizer=tokenizer,
        #                                     total_seq_len=args.max_ou15_text_len,
        #                                     text_max_sep_len=args.utt_max_len,
        #                                     filetype='train')

   
     
      

        relations = train_dataset_mol.relations | train_dataset_mol.relations
        type2ids, id2types = DialogueDataset.format_relations(relations)
        train_dataset_mol.get_relations(relations, type2ids, id2types)
        train_dataset_mol.get_discourse_graph()

        eval_dataset_mol.get_relations(relations, type2ids, id2types)
        eval_dataset_mol.get_discourse_graph()

        train_dataset_hu_ar.get_relations(relations, type2ids, id2types)
        train_dataset_hu_ar.get_discourse_graph()

        eval_dataset_hu_ar.get_relations(relations, type2ids, id2types)
        eval_dataset_hu_ar.get_discourse_graph()


        train_dataset_hu_si.get_relations(relations, type2ids, id2types)
        train_dataset_hu_si.get_discourse_graph()

        eval_dataset_hu_si.get_relations(relations, type2ids, id2types)
        eval_dataset_hu_si.get_discourse_graph()

        # train_dataset_ou5.get_relations(relations, type2ids, id2types)
        # train_dataset_ou5.get_discourse_graph()

        # train_dataset_ou10.get_relations(relations, type2ids, id2types)
        # train_dataset_ou10.get_discourse_graph()

        # train_dataset_ou15.get_relations(relations, type2ids, id2types)
        # train_dataset_ou15.get_discourse_graph()

        print('saving dataset..')
        torch.save(train_dataset_mol, train_mol_file)
        torch.save(eval_dataset_mol, eval_mol_file)
        torch.save(train_dataset_hu_ar, train_hu_ar_file)
        torch.save(eval_dataset_hu_ar, eval_hu_ar_file)
        torch.save(train_dataset_hu_si, train_hu_si_file)
        torch.save(eval_dataset_hu_si, eval_hu_si_file)
        torch.save(train_dataset_hu_rs, train_hu_rs_file)
        torch.save(eval_dataset_hu_rs, eval_hu_rs_file)
        # torch.save(train_dataset_ou5, train_ou5_file)
        # torch.save(train_dataset_ou10, train_ou10_file)
        # torch.save(train_dataset_ou15, train_ou15_file)
      
        # torch.save(train_dataset_ou5_rs, train_ou5_rs_file)
        # torch.save(train_dataset_ou10_rs, train_ou10_rs_file)
        # torch.save(train_dataset_ou15_rs, train_ou15_rs_file)
        
    args.relation_type_num = len(id2types)
    # pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)

    def train_collate_fn_mol(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.mol_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, speaker_ids,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                speaker_ids = torch.stack(speaker_ids, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, speaker_ids, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)

    def eval_collate_fn_mol(examples):
        texts, input_mask, segment_ids, speaker_ids, sep_index,pairs, graphs, speakers, turns, edu_nums, ids = zip(*examples)
        texts = torch.stack(texts, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        speaker_ids = torch.stack(speaker_ids, dim=0)
        assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == speaker_ids.shape[0] == len(sep_index)
        speakers = ints_to_tensor(list(speakers))
        turns = ints_to_tensor(list(turns))
        graphs = ints_to_tensor(list(graphs))
        edu_nums = torch.tensor(edu_nums)
        return texts, input_mask, segment_ids, speaker_ids, sep_index, pairs,graphs, speakers, turns, edu_nums, list(ids)

    def train_collate_fn_hu(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.hu_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, speaker_ids, sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                speaker_ids = torch.stack(speaker_ids, dim=0)
                
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] ==speaker_ids.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))

                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, speaker_ids, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)

    def train_collate_fn_ou_len5(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.ou5_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, _,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)

    def train_collate_fn_ou_len10(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.ou10_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, _,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)

    def train_collate_fn_ou_len15(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.ou15_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, _,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)


    def MultiTaskLearning(mtl_model, train_hu_ar_dataloader=None, train_ou5_ar_dataloader=None,
                                        train_ou10_ar_dataloader=None, train_ou15_ar_dataloader=None, 
                                        train_hu_si_dataloader=None, train_ou5_si_dataloader=None,
                                        train_ou10_si_dataloader=None, train_ou15_si_dataloader=None, 
                                        train_hu_rs_dataloader=None, train_ou5_rs_dataloader=None,
                                        train_ou10_rs_dataloader=None, train_ou15_rs_dataloader=None,
                                        train_mol_dataloader=None):
        step = 0
        total_mol_parsing_link_loss = total_hu_ar_loss = total_hu_si_loss = total_hu_rs_loss  = 0
        total_mol_parsing_Rel_loss  = 0
      
        print('training hu ar-------------')
        # #train hu ar
        for hu_data_batch in tqdm(train_hu_ar_dataloader):
            hu_ar_loss, _ = \
                mtl_model.train_minibatch('hu_ar', hu_data_batch)
            total_hu_ar_loss += hu_ar_loss
            step += 1
            if step % args.report_step == 0:
                print('\t{} step hu ar loss: {:.4f} '.format(step, total_hu_ar_loss / args.report_step))
                total_hu_ar_loss = 0
            if args.debug:
                break 
        print('training hu si-------------')
        # #train hu si
        for hu_data_batch in tqdm(train_hu_si_dataloader):
            hu_si_loss, _ = \
                mtl_model.train_minibatch('hu_si', hu_data_batch)
            total_hu_si_loss += hu_si_loss
            step += 1
            if step % args.report_step == 0:
                print('\t{} step hu ar loss: {:.4f} '.format(step, total_hu_si_loss / args.report_step))
                total_hu_si_loss = 0
            if args.debug:
                break 
        print('training hu rs-------------')
        # train hu RS
        for hu_rs_data_batch in tqdm(train_hu_rs_dataloader):
            hu_rs_link_loss, _ = \
                mtl_model.train_minibatch('hu_rs', hu_rs_data_batch, withSpkembedding=True)
            total_hu_rs_loss += hu_rs_link_loss
            step += 1
            if step % args.report_step == 0:
                print('\t{} step hu rs loss: {:.4f} '.format(step, total_hu_rs_loss / args.report_step))
                total_hu_rs_loss = 0
            if args.debug:
                break 
        
        print('training parsing-------------')
        # train mol
        for i in range(args.TrainingParsingTimes):
            for mol_data_batch in tqdm(train_mol_dataloader):
                temp_link_mol_loss, temp_rel_mol_loss = \
                    mtl_model.train_minibatch('parsing', mol_data_batch)
                total_mol_parsing_link_loss += temp_link_mol_loss
                total_mol_parsing_Rel_loss += temp_rel_mol_loss
                step += 1
                if step % 1 == 0:
                    print(
                        '\t{} mol link loss {:.4f}, rel loss {:.4f} '.format(step,
                                    total_mol_parsing_link_loss / args.report_step,
                                    total_mol_parsing_Rel_loss / args.report_step))
                    total_mol_parsing_link_loss = total_mol_parsing_Rel_loss = 0
                if args.debug:
                    break 
            
    def generate_TST_mask(args, model, task_type, train_dataloader, mask_save_path, withSpkEmbedding=False):
        if os.path.exists(mask_save_path) and not args.remake_mask:
            print('loading mask: {}'.format(mask_save_path))
            gradient_mask = torch.load(mask_save_path)
        else:
            
            gradient_mask = dict()
            model.train()
            for name, params in model.named_parameters():
                if 'SSAModule.gnn' in name :
                    gradient_mask[params] = params.new_zeros(params.size())
            N = len(train_dataloader)
            for batch in tqdm(train_dataloader):
                for mini_batch in batch:
                    texts, input_mask, segment_ids, speaker_ids, sep_index, pairs, graphs, speakers, turns, edu_nums = mini_batch
                    texts, input_mask, segment_ids, speaker_ids, graphs, speakers, turns, edu_nums = \
                        texts.cuda(), input_mask.cuda(), segment_ids.cuda(), speaker_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
                    mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
                    link_scores, label_scores = model.critic.task_output(task_type, texts, input_mask, segment_ids, speaker_ids,
                                                                        sep_index,
                                                                        edu_nums, speakers, turns,
                                                                        withSpkembedding=withSpkEmbedding)
                    
                    if task_type == 'hu_ar' or task_type == 'ou5_ar' or task_type == 'ou10_ar' or task_type == 'ou15_ar' or \
                        task_type == 'hu_si' or task_type == 'ou5_si' or task_type == 'ou10_si' or task_type == 'ou15_si' or task_type == 'parsing':
                        link_loss, label_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask)
                        link_loss = link_loss.mean()
                        label_loss = label_loss.mean()
                        if task_type=='parsing':
                            loss = link_loss + label_loss
                        else:
                            loss = link_loss
        
                    elif task_type == 'hu_rs' or task_type =='ou5_rs' or task_type =='ou10_rs' or task_type =='ou15_rs':
                        criter = nn.CrossEntropyLoss()
                        loss = criter(link_scores, graphs)
                    loss.backward()
                    for name, params in model.named_parameters():
                        if 'SSAModule.gnn' in name :
                            torch.nn.utils.clip_grad_norm_(params, 1.0)
                            gradient_mask[params] += (params.grad ** 2) / N
                    model.critic.task_model.zero_grad()
                    if args.debug:
                        break
                if args.debug:
                    break
            r = None
            for k, v in gradient_mask.items():
                v = v.view(-1).cpu().numpy()
                if r is None:
                    r = v
                else:
                    r = np.append(r, v)
            polar = np.percentile(r, args.alpha * 100)
            for k in gradient_mask:
                gradient_mask[k] = gradient_mask[k] >= polar
            torch.save(gradient_mask, mask_save_path)
            print('mask saved path: {}'.format(mask_save_path))
        return gradient_mask

    def generate_TSTAndBERT_mask(args, model, task_type, train_dataloader, mask_save_path, withSpkEmbedding=False):
        if os.path.exists(mask_save_path) and not args.remake_mask:
            print('loading mask: {}'.format(mask_save_path))
            gradient_mask = torch.load(mask_save_path)
        else:
            
            gradient_mask = dict()
            model.train()
            for name, params in model.named_parameters():
                if 'SSAModule.gnn' in name or 'pretrained_model.embeddings.' in name:
                    gradient_mask[params] = params.new_zeros(params.size())
            N = len(train_dataloader)
            for batch in tqdm(train_dataloader):
                for mini_batch in batch:
                    texts, input_mask, segment_ids, speaker_ids, sep_index, pairs, graphs, speakers, turns, edu_nums = mini_batch
                    texts, input_mask, segment_ids, speaker_ids, graphs, speakers, turns, edu_nums = \
                        texts.cuda(), input_mask.cuda(), segment_ids.cuda(), speaker_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
                    mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
                    link_scores, label_scores = model.critic.task_output(task_type, texts, input_mask, segment_ids, speaker_ids,
                                                                        sep_index,
                                                                        edu_nums, speakers, turns,
                                                                        withSpkembedding=withSpkEmbedding)
                    
                    if task_type == 'hu_ar' or task_type == 'ou5_ar' or task_type == 'ou10_ar' or task_type == 'ou15_ar' or \
                        task_type == 'hu_si' or task_type == 'ou5_si' or task_type == 'ou10_si' or task_type == 'ou15_si' or task_type == 'parsing':
                        link_loss, label_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask)
                        link_loss = link_loss.mean()
                        label_loss = label_loss.mean()
                        if task_type=='parsing':
                            loss = link_loss + label_loss
                        else:
                            loss = link_loss
        
                    elif task_type == 'hu_rs' or task_type =='ou5_rs' or task_type =='ou10_rs' or task_type =='ou15_rs':
                        criter = nn.CrossEntropyLoss()
                        loss = criter(link_scores, graphs)
                    loss.backward()
                    for name, params in model.named_parameters():
                        if 'SSAModule.gnn' in name or 'pretrained_model.embeddings.' in name:
                            torch.nn.utils.clip_grad_norm_(params, 1.0)
                            gradient_mask[params] += (params.grad ** 2) / N
                    model.critic.task_model.zero_grad()
                    if args.debug:
                        break
                if args.debug:
                    break
            r = None
            for k, v in gradient_mask.items():
                v = v.view(-1).cpu().numpy()
                if r is None:
                    r = v
                else:
                    r = np.append(r, v)
            polar = np.percentile(r, args.alpha * 100)
            for k in gradient_mask:
                gradient_mask[k] = gradient_mask[k] >= polar
            torch.save(gradient_mask, mask_save_path)
            print('mask saved path: {}'.format(mask_save_path))
        return gradient_mask

    def generate_TST_mask_all(args, model, task_type, train_dataloader):
        gradient_mask = dict()
        model.train()
        for name, params in model.named_parameters():
            if 'SSAModule.gnn' in name:
                gradient_mask[params] = params.new_zeros(params.size())
            if 'encoder.layer.' in name  in name :
                gradient_mask[params] = params.new_zeros(params.size())
        N = len(train_dataloader)
        for batch in tqdm(train_dataloader):
            for mini_batch in batch:
                texts, input_mask, segment_ids, labels, sep_index, pairs, graphs, speakers, turns, edu_nums = mini_batch
                texts, input_mask, segment_ids, graphs, speakers, turns, edu_nums = \
                    texts.cuda(), input_mask.cuda(), segment_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
                mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
                link_scores, label_scores = model.critic.task_output(task_type, texts, input_mask, segment_ids,
                                                                    sep_index,
                                                                    edu_nums, speakers, turns)
                link_loss, label_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask)
                link_loss = link_loss.mean()
                label_loss = label_loss.mean()
                if task_type == 'hu_ar' or task_type == 'ou5_ar' or task_type == 'ou10_ar' or task_type == 'ou15_ar':

                    loss = link_loss
                elif task_type == 'parsing':
                    loss = link_loss + label_loss
                loss.backward()
                for name, params in model.named_parameters():
                     if 'SSAModule.gnn' in name or 'encoder.layer.' in name:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        gradient_mask[params] += (params.grad ** 2) / N
                model.critic.task_model.zero_grad()
                if args.debug:
                    break
            if args.debug:
                break
        r = None
        for k, v in gradient_mask.items():
            v = v.view(-1).cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)
        polar = np.percentile(r, args.alpha * 100)
        for k in gradient_mask:
            gradient_mask[k] = gradient_mask[k] >= polar
        
        return gradient_mask
    
    eps = np.finfo(np.float32).eps.item()
   
    
    if args.do_train:
        train_dataloader_mol = DataLoader(dataset=train_dataset_mol, batch_size=args.mol_pool_size,
                                          shuffle=True,
                                          collate_fn=train_collate_fn_mol)

        eval_dataloader_mol = DataLoader(dataset=eval_dataset_mol, batch_size=args.eval_mol_pool_size,
                                         shuffle=False,
                                         collate_fn=eval_collate_fn_mol)

        train_dataloader_hu_ar = DataLoader(dataset=train_dataset_hu_ar, batch_size=args.hu_pool_size, shuffle=True,
                                            collate_fn=train_collate_fn_mol)
        
        eval_dataloader_hu_ar = DataLoader(dataset=eval_dataset_hu_ar, batch_size=args.hu_pool_size, shuffle=False,
                                            collate_fn=eval_collate_fn_mol)

        train_dataloader_hu_si = DataLoader(dataset=train_dataset_hu_si, batch_size=args.hu_pool_size, shuffle=True,
                                            collate_fn=train_collate_fn_mol)
        
        eval_dataloader_hu_si = DataLoader(dataset=eval_dataset_hu_si, batch_size=args.hu_pool_size, shuffle=False,
                                            collate_fn=eval_collate_fn_mol)
        train_dataloader_hu_rs =   DataLoader(dataset=train_dataset_hu_rs, batch_size=args.hu_pool_size,
                                               shuffle=True,
                                               collate_fn=train_collate_fn_mol)
      
        eval_dataloader_hu_rs =   DataLoader(dataset=eval_dataset_hu_rs, batch_size=args.hu_pool_size,
                                               shuffle=False,
                                               collate_fn=eval_collate_fn_mol)
        # train_dataloader_hu = ''  

        # train_dataloader_ou5 = DataLoader(dataset=train_dataset_ou5, batch_size=args.ou5_pool_size,
        #                                        shuffle=True,
        #                                        collate_fn=train_collate_fn_ou_len5)

        # train_dataloader_ou10 = DataLoader(dataset=train_dataset_ou10, batch_size=args.ou10_pool_size,
        #                                        shuffle=True,
        #                                        collate_fn=train_collate_fn_ou_len10)

        # train_dataloader_ou15 = DataLoader(dataset=train_dataset_ou15, batch_size=args.ou15_pool_size,
        #                                        shuffle=True,
        #                                        collate_fn=train_collate_fn_ou_len15)
        train_dataloader_ou10 = ''
        train_dataloader_ou15 = ''
        #TST
        
        pretrained_model = BertWithSpeakerID(args) # bert_model_name, speaker_id_dim, num_speakers 
        model = PolicyNetwork(args=args, pretrained_model=pretrained_model)
        model = model.to(args.device)
        if args.TST_Learning_Mode:
            print('begin generate task mask')
            state_dict = torch.load(args.ST_model_path+'.pt')
            model.load_state_dict(state_dict, strict=False)
            parsing_mask = generate_TST_mask(args, model, 'parsing', train_dataloader_mol, mask_save_path=args.mol_parsing_mask_path)
            hu_ar_mask = generate_TST_mask(args, model, 'hu_ar', train_dataloader_hu_ar, mask_save_path=args.hu_ar_mask_path)
            hu_si_mask = generate_TST_mask(args, model, 'hu_si', train_dataloader_hu_si, mask_save_path=args.hu_si_mask_path)
            hu_rs_mask = generate_TST_mask(args, model, 'hu_rs', train_dataloader_hu_rs, mask_save_path=args.hu_rs_mask_path, withSpkEmbedding=True)

            model.set_gradient_mask(parsing_mask, 'parsing')
            model.set_gradient_mask(hu_ar_mask, 'hu_ar')
            model.set_gradient_mask(hu_si_mask, 'hu_si')
            model.set_gradient_mask(hu_rs_mask, 'hu_rs')
        # ou5_mask = generate_TST_mask_all(args, model, 'ou5_ar', train_dataloader_ou5)
        # ou10_mask = generate_TST_mask(args, model, 'ou10_ar', train_dataloader_ou10)
        # ou15_mask = generate_TST_mask(args, model, 'ou15_ar', train_dataloader_ou15)
        
        # model.set_gradient_mask(parsing_mask, 'parsing')
        # model.set_gradient_mask(hu_mask, 'hu_ar')
        # model.set_gradient_mask(ou5_mask, 'ou5_ar')
        # model.set_gradient_mask(ou10_mask, 'ou10_ar')
        # model.set_gradient_mask(ou15_mask, 'ou15_ar')
        if args.TST_Learning_Mode:
            print('begin training TST')
        else:
            print('begin training ST')

        max_reward = 1000
        max_epoch = -1
        if args.TST_Learning_Mode:
            total_epoch = args.TST_epoches
        else:
            total_epoch = args.ST_epoches
        
        for epoch in range(total_epoch):
            # print('{} epoch TST finetuning..'.format(epoch + 1))
            model.train() 
            MultiTaskLearning(model, 
                              train_mol_dataloader = train_dataloader_mol,
                              train_hu_ar_dataloader = train_dataloader_hu_ar,
                              train_hu_si_dataloader = train_dataloader_hu_si,
                              train_hu_rs_dataloader = train_dataloader_hu_rs 
                              )
            

            mol_linkandrel_loss, _ = model.compute_f1_and_loss_reward(tasktype='parsing',
                                                                      eval_dataloader=eval_dataloader_mol)
           
            Pat1_ar, SessAc_ar, ar_link_loss =  model.compute_Pat1_and_loss_reward(tasktype='hu_ar',
                                                                      eval_dataloader=eval_dataloader_hu_ar,
                                                                      source_file=args.eval_hu_ar_file)

            hu_rs_eval_loss, hu_epoch_f1 = model.compute_RS_f1_and_loss_reward(tasktype='hu_rs',
                                                                      eval_dataloader=eval_dataloader_hu_rs)
            
            

            Pat1_si, hu_si_loss = model.compute_SI_Pat1_and_loss_reward(tasktype='hu_si',
                                                                      eval_dataloader=eval_dataloader_hu_si,
                                                                      source_file=args.eval_hu_si_file)
            
           
            
            total_loss = mol_linkandrel_loss + mol_linkandrel_loss + hu_rs_eval_loss + hu_si_loss + ar_link_loss
            # total_loss =hu_rs_eval_loss


            if total_loss < max_reward:
                if args.TST_Learning_Mode:
                    torch.save(model.state_dict(), args.TST_model_path + '.pt')
                else:
                    torch.save(model.state_dict(), args.ST_model_path + '.pt')
                max_reward = total_loss
                max_epoch = epoch
            # print('eval hu rs eval loss {}'.format(hu_rs_eval_loss))
         

    else:
        test_dataloader_mol = DataLoader(dataset=test_dataset_mol, batch_size=args.eval_mol_pool_size,
                                         shuffle=False,
                                         collate_fn=eval_collate_fn_mol)

        test_dataloader_hu_rs =   DataLoader(dataset=test_dataset_hu_rs, batch_size=args.hu_pool_size,
                                               shuffle=False,
                                               collate_fn=eval_collate_fn_mol)

        test_dataloader_hu_ar = DataLoader(dataset=test_dataset_hu_ar, batch_size=args.hu_pool_size, shuffle=False,
                                            collate_fn=eval_collate_fn_mol)
        test_dataloader_hu_si = DataLoader(dataset=test_dataset_hu_si, batch_size=args.hu_pool_size, shuffle=False,
                                            collate_fn=eval_collate_fn_mol)
        
       
        pretrained_model = BertWithSpeakerID(args) # bert_model_name, speaker_id_dim, num_speakers 
        model = PolicyNetwork(args=args, pretrained_model=pretrained_model)
        model = model.to(args.device)
        if args.TST_Learning_Mode:
            print('loading path : {}'.format(args.TST_model_path+'.pt'))
            state_dict = torch.load(args.TST_model_path+'.pt')
        else:
            print('loading path : {}'.format(args.ST_model_path+'.pt'))
            state_dict = torch.load(args.ST_model_path+'.pt')
        model.load_state_dict(state_dict,strict=False)
        model.eval()
        print('evaluating Mol Parsing:')
        total_loss, total_f1 = model.compute_f1_and_loss_reward(tasktype='parsing',
                                                          eval_dataloader=test_dataloader_mol)
        print('evaluating hu ar:')
        Pat1_ar, SessAc_ar, ar_link_loss =  model.compute_Pat1_and_loss_reward(tasktype='hu_ar',
                                                                      eval_dataloader=test_dataloader_hu_ar,
                                                                      source_file=args.test_hu_ar_file)

        print('evaluating hu si :')
        Pat1, _ = model.compute_SI_Pat1_and_loss_reward(tasktype='hu_si',
                                                          eval_dataloader=test_dataloader_hu_si,
                                                           source_file=args.test_hu_si_file)
        print('evaluating hu rs:')
        hu_rs_eval_loss, hu_epoch_f1 = model.compute_RS_f1_and_loss_reward(tasktype='hu_rs',
                                                            eval_dataloader=test_dataloader_hu_rs)
