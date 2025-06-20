from .base import AbstractDataloader
from .utils import *

import torch
import torch.utils.data as data_utils

import random
import numpy as np
import os
import pickle
import math
from py2neo import Graph, Node, Relationship
import re

import transformers
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
from trainer import absolute_recall_mrr_ndcg_for_ks


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()


def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)                                                      
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# the following prompting is based on alpaca
def generate_and_tokenize_eval(args, data_point, tokenizer, prompter,
                               user=None, probs=None, index=None):
    in_prompt = prompter.generate_prompt(data_point["system"],
                                         data_point["input"])

    tokenized_full_prompt = tokenizer(in_prompt,
                                      truncation=True,
                                      max_length=args.llm_max_text_len,
                                      padding=False,
                                      return_tensors=None)
    
    tokenized_full_prompt["labels"] = ord(data_point["output"]) - ord('A')
    
    return tokenized_full_prompt


def generate_and_tokenize_train(args, data_point, tokenizer, prompter,
                                user=None, probs=None, index=None):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt,
                           truncation=True,
                           max_length=args.llm_max_text_len,
                           padding=False,
                           return_tensors=None)
        if (result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    full_prompt = prompter.generate_prompt(data_point["system"],
                                           data_point["input"],
                                           data_point["output"])
    
    tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
    
    if not args.llm_train_on_inputs:
        tokenized_full_prompt["labels"][:-2] = [-100] * len(tokenized_full_prompt["labels"][:-2])
    
    return tokenized_full_prompt


def seq_to_token_ids(args, seq, candidates, label, text_dict, tokenizer, prompter, eval=False,
                    index=None, user=None, relations=None, probs=None):
    def truncate_title(title):
        if "," in title:
            title = title.split(",")[0]
        title_ = tokenizer.tokenize(title)[:args.llm_max_title_len]
        title = tokenizer.convert_tokens_to_string(title_)
        return title

    seq_t = ' \n '.join(['(' + str(idx + 1) + ') ' + truncate_title(text_dict[item]) 
                       for idx, item in enumerate(seq)])
    can_t = ' \n '.join(['(' + chr(ord('A') + idx) + ') ' + truncate_title(text_dict[item])
                       for idx, item in enumerate(candidates)])
    output = chr(ord('A') + candidates.index(label))  # ranking only
    
    if not args.llm_train_with_relation:
        data_point = {}
        data_point['system'] = args.llm_system_template if args.llm_system_template is not None else DEFAULT_SYSTEM_PROMPT
        data_point['input'] = args.llm_input_template.format(seq_t, can_t)
        data_point['output'] = output
    else:
        data_point = {}
        data_point['system'] = args.llm_system_template_relation if args.llm_system_template_relation is not None else DEFAULT_SYSTEM_PROMPT
        data_point['input'] = args.llm_input_template_relation.format(seq_t, relations, can_t)
        data_point['output'] = output
    
    if eval:
        return generate_and_tokenize_eval(args, data_point, tokenizer, prompter, user, probs, index)
    else:
        return generate_and_tokenize_train(args, data_point, tokenizer, prompter, user, probs, index)


def extract_relations(args, graph, user, seq, candidates, answer, text_dict,
                      tokenizer, probs, target_as_candidate=False):
    def truncate_title(title):
        if "," in title:
            title = title.split(",")[0]
        title_ = tokenizer.tokenize(title)[:args.llm_max_title_len]
        title = tokenizer.convert_tokens_to_string(title_)
        return title
    
    seq_l = ['(' + str(idx + 1) + ') ' + truncate_title(text_dict[item]) 
                       for idx, item in enumerate(seq)]
    can_l = ['(' + chr(ord('A') + idx) + ') ' + truncate_title(text_dict[item])
                       for idx, item in enumerate(candidates)]
    
    if target_as_candidate == True:
        output = chr(ord('A') + candidates.index(answer))
        candidates = [answer]
        can_l = ['(' + output + ') ' + truncate_title(text_dict[item])
                           for idx, item in enumerate(candidates)]

    try:
        probs = probs.tolist()[0]
    except:
        pass

    # Querying the relationships from the graph
    relations = []
    for h_idx, h_id in enumerate(seq):
        # Extracting the shortest path from the user to the item in the user's history
        try:
            res = list(graph.run(QUERIES[args.dataset_code]["q_u_h"], u_id=str(user), i_id=str(h_id)).data()[0].values())
        except:
            continue

        res_u_h = "The user rated the " + str(seq_l[h_idx]) + " " + str(res[2][0]) + " stars. "

        temp_relations = []
        for c_idx, c_id in enumerate(candidates): 
            # Extracting the shortest path from an item in the user's history to an item in the user's candidate pool
            try:
                res = list(graph.run(QUERIES[args.dataset_code]["q_h_c"], h_id=str(h_id), i_id=str(c_id)).data()[0].values())
            except:
                continue
            
            if args.dataset_code == "ml-100k":
                res = [item for sublist in zip(res[0], res[1]) for item in sublist] + [res[0][-1]]
                res[0] = seq_l[h_idx]
                res[-1] = can_l[c_idx]
                for res_idx, res_itm in enumerate(res):
                    try:
                        if res_idx - 1 >= 0 and "RELEASED_YEAR_IS" in res[res_idx-1]:
                            continue
                        res[res_idx] = truncate_title(text_dict[int(res_itm)])
                    except:
                        res[res_idx] = res_itm
            elif args.dataset_code == "beauty":
                res = [item for sublist in zip(res[0], res[1]) for item in sublist] + [res[0][-1]]
                res[0] = seq_l[h_idx]
                res[-1] = can_l[c_idx]
                for res_idx, res_itm in enumerate(res):
                    try:
                        res[res_idx] = truncate_title(text_dict[int(res_itm)])
                    except:
                        res[res_idx] = res_itm
            
            res = " -> ".join(res)
            temp_relations.append(res_u_h + res)

        if not temp_relations:
            continue
            
        if probs:
            N = len(temp_relations)
            df_raw = [0] * len(args.llm_relations_class)
            for item in temp_relations:
                for cls_idx, cls_name in enumerate(args.llm_relations_class):
                    df_raw[cls_idx] += item.count(cls_name)
            df = [math.log( N / (item+1) ) + 1 for item in df_raw]
            scores = []
            for item in temp_relations:
                tf = [0] * len(args.llm_relations_class)
                for cls_idx, cls_name in enumerate(args.llm_relations_class):
                    tf[cls_idx] = item.count(cls_name)
                tf = [item/sum(tf) for item in tf]
                tf = [item * item_df for item, item_df in zip(tf, df)]
                tf = softmax(tf)
                tf_idf = [float(x * y) for x, y in zip(tf, probs)]
                scores.append(sum(tf_idf))
            max_idx = scores.index(max(scores))
            res_u_c = temp_relations[max_idx]
        else:
            count_relations = [item.count("->") for item in temp_relations]
            min_idx = count_relations.index(min(count_relations))
            res_u_c = temp_relations[min_idx]
        
        relations.append(res_u_c)

    relations = " \n ".join(relations)
    
    return relations


class LLMDataloader():
    def __init__(self, args, dataset, model):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        seq_dataset = dataset.load_dataset()
        self.train = seq_dataset['train']
        self.val = seq_dataset['val']
        self.test = seq_dataset['test']
        self.umap = seq_dataset['umap']
        self.smap = seq_dataset['smap']
        self.text_dict = seq_dataset['meta']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        
        args.num_items = self.item_count
        self.max_len = args.llm_max_history
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_base_tokenizer, cache_dir=args.llm_cache_dir)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.clean_up_tokenization_spaces = True
        self.prompter = Prompter()
        
        self.llm_retrieved_path = args.llm_retrieved_path
        print('Loading retrieved file from {}'.format(self.llm_retrieved_path))
        retrieved_file = pickle.load(open(os.path.join(args.llm_retrieved_path,
                                                       'retrieved.pkl'), 'rb'))
        
        self.graph = None
        if args.llm_train_with_relation:
            graph = Graph(args.neo4j_uri, auth=(args.neo4j_user_name, args.neo4j_password))
            self.graph = graph
        
        self.user_preferences_model = model
        
        print('******************** Constructing Validation Subset ********************')
        self.val_probs = retrieved_file['val_probs']
        self.val_labels = retrieved_file['val_labels']
        self.val_metrics = retrieved_file['val_metrics']
        self.val_users = [u for u, (p, l) in enumerate(zip(self.val_probs, self.val_labels), start=1) \
                          if l in torch.topk(torch.tensor(p), self.args.llm_negative_sample_size+1).indices]
        self.val_candidates = [torch.topk(torch.tensor(self.val_probs[u-1]), 
                                self.args.llm_negative_sample_size+1).indices.tolist() for u in self.val_users]

        print('******************** Constructing Test Subset ********************')
        self.test_probs = retrieved_file['test_probs']
        self.test_labels = retrieved_file['test_labels']
        self.test_metrics = retrieved_file['test_metrics']
        self.test_users = [u for u, (p, l) in enumerate(zip(self.test_probs, self.test_labels), start=1) \
                          if l in torch.topk(torch.tensor(p), self.args.llm_negative_sample_size+1).indices]
        self.test_candidates = [torch.topk(torch.tensor(self.test_probs[u-1]), 
                                self.args.llm_negative_sample_size+1).indices.tolist() for u in self.test_users]
        self.non_test_users = [u for u, (p, l) in enumerate(zip(self.test_probs, self.test_labels), start=1) \
                               if l not in torch.topk(torch.tensor(p), self.args.llm_negative_sample_size+1).indices]
        self.test_retrieval = {
            'original_size': len(self.test_probs),
            'retrieval_size': len(self.test_candidates),
            'original_metrics': self.test_metrics,
            'retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.test_probs)[torch.tensor(self.test_users)-1],
                torch.tensor(self.test_labels)[torch.tensor(self.test_users)-1],
                self.args.metric_ks,
            ),
            'non_retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.test_probs)[torch.tensor(self.non_test_users)-1],
                torch.tensor(self.test_labels)[torch.tensor(self.non_test_users)-1],
                self.args.metric_ks,
            ),
        }

    @classmethod
    def code(cls):
        return 'llm'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.lora_micro_batch_size,
                                           shuffle=True, pin_memory=True, num_workers=self.args.num_workers,
                                           worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = LLMTrainDataset(self.args, self.train, self.max_len, self.rng,
                                  self.text_dict, self.tokenizer, self.prompter,
                                  self.graph, self.user_preferences_model)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = LLMValidDataset(self.args, self.train, self.val, self.max_len, self.rng,
                                      self.text_dict, self.tokenizer, self.prompter, self.val_users,
                                      self.val_candidates, self.graph, self.user_preferences_model)
        elif mode == 'test':
            dataset = LLMTestDataset(self.args, self.train, self.val, self.test, self.max_len,
                                     self.rng, self.text_dict, self.tokenizer, self.prompter, self.test_users,
                                     self.test_candidates, self.graph, self.user_preferences_model)
        return dataset


class LLMTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len, rng, text_dict, tokenizer, prompter, graph, model):
        self.args = args
        self.max_len = max_len
        self.num_items = args.num_items
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.graph = graph
        self.model = model

        self.all_seqs = []
        self.all_users = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            for i in range(2, len(seq)+1):
                self.all_seqs += [seq[:i]]
                self.all_users.append(u)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index]
        answer = tokens[-1]
        original_seq = tokens[:-1]
        
        seq = original_seq[-self.max_len:]
        cur_idx, candidates = 0, [answer]
        samples = self.rng.randint(1, self.args.num_items+1, size=5*self.args.llm_negative_sample_size)
        while len(candidates) < self.args.llm_negative_sample_size + 1:
            item = samples[cur_idx]
            cur_idx += 1
            if item in original_seq or item == answer: continue
            else: candidates.append(item)
        self.rng.shuffle(candidates)

        user = self.all_users[index]
        probs, relations = None, None
        if self.args.llm_train_with_relation:
            if self.args.llm_train_with_relation_score:
                user_vec = torch.LongTensor([user])
                probs = self.model.user_preferences_forward(user_vec)
            relations = extract_relations(self.args, self.graph, user, seq, candidates, answer,
                                          self.text_dict, self.tokenizer, probs,
                                          target_as_candidate=False)

        return seq_to_token_ids(self.args, seq, candidates, answer, self.text_dict,
                                self.tokenizer, self.prompter, eval=False,
                                index=index, user=user, relations=relations, probs=probs)


class LLMValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, max_len, rng, text_dict,
                 tokenizer, prompter, val_users, val_candidates, graph, model):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.val_users = val_users
        self.val_candidates = val_candidates
        self.graph = graph
        self.model = model

    def __len__(self):
        return len(self.val_users)

    def __getitem__(self, index):
        user = self.val_users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user][0]
        
        seq = seq[-self.max_len:]
        candidates = self.val_candidates[index]
        assert answer in candidates
        # self.rng.shuffle(candidates)
        
        probs, relations = None, None
        if self.args.llm_train_with_relation:
            if self.args.llm_train_with_relation_score:
                user_vec = torch.LongTensor([user])
                self.model.eval()
                with torch.no_grad():
                    probs = self.model.user_preferences_forward(user_vec)
                self.model.train()
            relations = extract_relations(self.args, self.graph, user, seq, candidates, answer,
                                          self.text_dict, self.tokenizer, probs,
                                          target_as_candidate=False)

        return seq_to_token_ids(self.args, seq, candidates, answer,
                                self.text_dict, self.tokenizer, self.prompter, eval=True,
                                index=index, user=user, relations=relations, probs=probs)


class LLMTestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2val, u2answer, max_len, rng, text_dict,
                 tokenizer, prompter, test_users, test_candidates, graph, model):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        self.users = sorted(u2seq.keys())
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.test_users = test_users
        self.test_candidates = test_candidates
        self.graph = graph
        self.model = model
    
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, index):
        user = self.test_users[index]
        seq = self.u2seq[user] + self.u2val[user]
        answer = self.u2answer[user][0]

        seq = seq[-self.max_len:]
        candidates = self.test_candidates[index]
        assert answer in candidates
        # self.rng.shuffle(candidates)

        probs, relations = None, None
        if self.args.llm_train_with_relation:
            if self.args.llm_train_with_relation_score:
                user_vec = torch.LongTensor([user])
                self.model.eval()
                with torch.no_grad():
                    probs = self.model.user_preferences_forward(user_vec)
                self.model.train()
            relations = extract_relations(self.args, self.graph, user, seq, candidates, answer,
                                          self.text_dict, self.tokenizer, probs,
                                          target_as_candidate=False)
            
        return seq_to_token_ids(self.args, seq, candidates, answer,
                                self.text_dict, self.tokenizer, self.prompter, eval=True,
                                index=index, user=user, relations=relations, probs=probs)