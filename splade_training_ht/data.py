from torch.utils.data import Dataset
import random
from sentence_transformers import InputExample
import torch
import numpy as np
# We create a custom MS MARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
ce_threshold = -3
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, ce_scores, num_neg = 1, loss_type = "marginmse", topk=20, model_type = "colbert"):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.num_neg = num_neg
        self.loss_type = loss_type
        if self.loss_type == "marginmse":
            assert(self.num_neg == 1)

        self.model_type = model_type
        '''
        for qid in self.queries:          
            pos_list = []
            for posid in self.queries[qid]['pos']:
                if posid in self.queries[qid]['neg']:
                    pos_list.append([self.queries[qid]['neg'].index(posid) + 1, posid])
                else:
                    pos_list.append([len(self.queries[qid]['neg']) + 1, posid])
            
            target_scores = [[pid[1], ce_scores[qid][pid[1]]] for pid in pos_list] + [[pid, ce_scores[qid][pid]] for pid in self.queries[qid]['neg']]
            target_scores = sorted(target_scores, key = lambda x: -x[1])
            target_ids = [x[0] for x in target_scores]
            
            self.queries[qid]['pos'] = [x + [target_ids.index(x[1]) + 1] for x in pos_list]
            self.queries[qid]['neg'] = [[x[0] + 1, x[1], target_ids.index(x[1]) + 1] for x in enumerate(self.queries[qid]['neg'])]
         '''
        for qid in self.queries:
            self.queries[qid]['neg'] = self.queries[qid]['neg'][:topk]
            random.shuffle(self.queries[qid]['neg'])
        self.iter_num = 0 

    def __getitem__(self, item):
        self.iter_num += 1
        query = self.queries[self.queries_ids[item]]
        if self.model_type == "colbert":
            query_text = "[unused0] " + query['query']
        else:
            query_text = query['query']

        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)   #Pop positive and add at end
            if self.model_type == "colbert":
                pos_text = "[unused1] " + self.corpus[pos_id[1]]
            else:
                pos_text = self.corpus[pos_id[1]]
            query['pos'].append(pos_id)
        else:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            if self.model_type == "colbert":
                pos_text = "[unused1] " + self.corpus[pos_id[1]]
            else:
                pos_text = self.corpus[pos_id[1]]
            query['neg'].append(pos_id)
        
        pos_score = self.ce_scores[qid][pos_id[1]]
        pos_idx = pos_id[0]
        pos_ce_idx = pos_id[2]
        #Get a negative passage
        neg_texts = []
        neg_scores = []
        neg_idx = []
        neg_ce_idx = []
        for i in range(self.num_neg):
            neg_id = query['neg'].pop(0)    #Pop negative and add at end
            if self.model_type == "colbert":
                neg_text = "[unused1] " + self.corpus[neg_id[1]]
            else:
                neg_text = self.corpus[neg_id[1]]
            if neg_id[1] in self.ce_scores[qid]:
                neg_score = self.ce_scores[qid][neg_id[1]]
            else:
                i -= 1
                continue
            query['neg'].append(neg_id)
            neg_texts.append(neg_text)
            neg_scores.append(neg_score)
            neg_idx.append(neg_id[0])
            neg_ce_idx.append(neg_id[2])

        
        
        if self.loss_type == "marginmse":
            return InputExample(texts=[query_text, pos_text, neg_texts[0]], label=pos_score-neg_scores[0])
        elif self.loss_type == "marginmse_ib":
            return InputExample(texts=[query_text, pos_text, neg_texts[0]], label=[pos_score,neg_scores[0]])
        elif self.loss_type in ["kldiv", "kldiv_focal", "kldiv_ib"]:
            target_score = torch.tensor([pos_score] + neg_scores)
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist()) # length of label is number of texts
        elif self.loss_type == "marginkldiv":
            target_score = torch.tensor([pos_score  - neg_score for neg_score in neg_scores])
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist()) # length of label is number 
        elif self.loss_type == "marginmse_position":
            return InputExample(texts=[query_text, pos_text, neg_texts[0]], label=[pos_score-neg_scores[0], pos_idx, neg_idx[0]])
        elif self.loss_type in ["wce", "ce"]:
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=[pos_idx] + neg_idx)
        elif self.loss_type == "marginkldiv_position":
            ce_diffs = [neg_score - pos_score for neg_score in neg_scores]
            target_score = torch.tensor(ce_diffs)
            target_score = torch.nn.functional.log_softmax(target_score)
            
            ##########weight defined 1 #############
            # alpha = 0.2
            # weights = [alpha/(1+np.exp(neg_i - pos_idx)) + 1 for neg_i in neg_idx]
            # eights = [w if ce_diff > ce_threshold else 1.0 for w,ce_diff in zip(weights,ce_diffs)]
            ##########weight define 2 ##############
            #weights = [np.log10(max(pos_idx - pos_ce_idx, 1))/2 + 1] * len(neg_idx)
            
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist() + [pos_idx] + neg_idx) # length of label is number 
        elif self.loss_type in ["kldiv_position", 'kldiv_position_focal']:
            ce_diffs = [neg_score - pos_score for neg_score in neg_scores]
            target_score = torch.tensor([pos_score] + neg_scores)
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist() + [pos_idx] + neg_idx) # length of label is number 
        elif self.loss_type == "kldiv_position_reverse":
            ce_diffs = [neg_score - pos_score for neg_score in neg_scores]
            target_score = -torch.tensor([pos_score] + neg_scores)
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist() + [pos_idx] + neg_idx) # length of label is number 
        
        else:
            raise("Unrecogized loss type!")
            return 

    def __len__(self):
        return len(self.queries)


class MSMARCODatasetCE(Dataset):
    def __init__(self, queries, corpus, ce_scores, num_neg = 1, topk=20, loss_type = "marginkldiv_position"):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.num_neg = num_neg
        self.loss_type = loss_type
       
      
        for qid in self.queries:
            self.queries[qid]['neg'] = self.queries[qid]['neg'][:topk]
            random.shuffle(self.queries[qid]['neg'])
        self.iter_num = 0 

    def __getitem__(self, item):
        self.iter_num += 1
        query = self.queries[self.queries_ids[item]]
        query_text = "[unused0] " + query['query']
        
        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)   #Pop positive and add at end
            pos_text = "[unused1] " + self.corpus[pos_id[1]]
            query['pos'].append(pos_id)
        else:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            pos_text = "[unused1] " + self.corpus[pos_id[1]]
            query['neg'].append(pos_id)
        
        pos_score = self.ce_scores[qid][pos_id[1]]
        pos_idx = pos_id[0]
        pos_ce_idx = pos_id[2]
        #Get a negative passage
        neg_texts = []
        neg_scores = []
        neg_idx = []
        neg_ce_idx = []

        for i in range(self.num_neg):
            neg_id = query['neg'].pop(0)    #Pop negative and add at end
            neg_text = "[unused1] " + self.corpus[neg_id[1]]
            query['neg'].append(neg_id)
            neg_texts.append(neg_text)
            neg_score = self.ce_scores[qid][neg_id[1]]
            neg_scores.append(neg_score)
            neg_idx.append(neg_id[0])
            neg_ce_idx.append(neg_id[2])
            
        
        neg_texts = [f"{query_text} [SEP] {neg_text}" for neg_text in neg_texts]
        pos_text = f"{query_text} [SEP] {pos_text}" 
        #alpha = 0.2
        #weights = [alpha/(1+np.exp(neg_i - pos_idx)) + 1 for neg_i in neg_idx]
        
        if self.loss_type == "crossentropy":   
            return InputExample(texts=[pos_text] + neg_texts) # length of label is number 
        elif self.loss_type == "marginkldiv_position":
            ce_diffs = [pos_score  - neg_score for neg_score in neg_scores]
            target_score = torch.tensor(ce_diffs)
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[pos_text] + neg_texts, label=target_score.tolist() + [pos_idx] + neg_idx)
        

    def __len__(self):
        return len(self.queries)



