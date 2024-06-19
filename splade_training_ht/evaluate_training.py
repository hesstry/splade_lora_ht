import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import os
import pytrec_eval
from collections import OrderedDict, defaultdict
from statistics import mean
import json
import sys
from models import MLMTransformerDense, ColBERTTransformer
from losses import pairwise_dot_score

import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, InputExample
from sbert import SentenceTransformerA
import models
import logging
from datetime import datetime
import gzip
import os
import tarfile
from torch.utils.data import Dataset
import random
from shutil import copyfile
import pickle
import argparse
import losses
import torch
from collections import defaultdict
from data import MSMARCODataset
import json
def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches
agg = "max"
bsize = 128


def evaluate_trainining(model, tokenizer, train_queries, corpus, k1, k2):
    fo = open(os.path.join(model_save_path, "train_queries.json"), "w")
    model.eval()
    with torch.no_grad():
        for q in tqdm(train_queries):
            dids = [x[1] for x in train_queries[q]['pos']] + [x[1] for x in train_queries[q]['neg']]
            dtexts = [corpus[did] for did in dids]

            q_features = tokenizer("[unused0] " + train_queries[q]['query'], return_tensors="pt").to('cuda')
            q_features = model(q_features)
            token_rep_q = torch.nn.functional.normalize(q_features['last_layer_embeddings'], p=2, dim=2)

            d_features = tokenizer(["[unused1] " + dtext for dtext in dtexts], return_tensors="pt", max_length=256,truncation=True,padding=True)
            d_features = _split_into_batches(d_features,bsize=bsize)
            all_scores = []
            for batch in d_features:
                d_batch = model(batch)
                d_mask = d_batch['attention_mask'].to('cuda')
                d_emb = d_batch['last_layer_embeddings']
                del d_batch
                d_mask = d_mask.unsqueeze(-1)
                token_rep_d =  d_emb * d_mask
                del d_mask, d_emb
                token_rep_d = torch.nn.functional.normalize(token_rep_d)
                scores =  (token_rep_q @ token_rep_d.permute(0,2,1)).max(2).values.sum(1).tolist()
                del token_rep_d
                torch.cuda.empty_cache()
                all_scores.extend(scores)

            reranking_results = sorted([[did,score] for did, score in zip(dids, all_scores)], key = lambda x: -x[1])
            reranking_results = {x[1][0]: 1/(int(x[0]) + 1 + k2) for x in enumerate(reranking_results)}

            retrieval_results = {x[1]: 1/(int(x[0]) + k1) for x in train_queries[q]['splade_pos']} | {x[1]: 1/(int(x[0]) + k1) for x in train_queries[q]['splade_neg']}

            combined_results = []
            for did in reranking_results:
                combined_results.append([did, reranking_results[did] + retrieval_results[did]])
            combined_results = sorted(combined_results, key = lambda x: -x[1])
            combined_lookup = dict()
            for x in enumerate(combined_results):
                combined_lookup[x[1][0]] = x[0] + 1

            pos_list = []
            for x in train_queries[q]['pos']:
                pos_list.append([combined_lookup[x[1]], x[1], x[2]])

            neg_list = []
            for x in train_queries[q]['neg']:
                neg_list.append([combined_lookup[x[1]], x[1], x[2]])

            train_queries[q]['pos'] = pos_list
            train_queries[q]['neg'] = neg_list
            
            fo.write(f"{q}\t{json.dumps(train_queries[q])}\n")
            fo.flush()
        fo.close()
     
    
                
if __name__ == "__main__":
    train_batch_size = 16  # Increasing the train batch size generally improves the model performance, but requires more GPU memory
    model_name = "output/colbert_g_num50_marginkldiv_position5-batch_size_8x4-2022-07-18_06-05-04/20000/0_ColBERTTransformer"
    max_passages = 0
    ce_score_margin = 0
    max_seq_length = 256  # Max length for passages. Increasing it implies more GPU memory needed
    num_negs_per_system = 50 # We used different systems to mine hard negatives. Number of hard negatives to add from each system

    # Load our embedding model

    word_embedding_model = models.ColBERTTransformer(model_name, max_seq_length=max_seq_length)
    print(len(word_embedding_model.tokenizer))
    tokens = ["[unused0]", "[unused1]", "[unused2]"] #[unused0] for query, [unused1] for doc, [unused2] for query expansion
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    print(len(word_embedding_model.tokenizer))

    checkpoint = torch.load(os.path.join(model_name, "checkpoint.pt"), map_location='cpu')
    word_embedding_model.load_state_dict(checkpoint)

    model = SentenceTransformerA(modules=[word_embedding_model])
    model_save_path = f'output/colbert_dynamic_fromwarmup_num50_marginkldiv_position5-batch_size_16-2022-07-13_22-18-35/num1' #_prfspladedoc310

    # Write self to path
    os.makedirs(model_save_path, exist_ok=True)

    train_script_path = os.path.join(model_save_path, 'train_script.py')

    ### Now we read the MS MARCO dataset

    data_folder = '../../msmarco'
    #data_folder = '/home/ec2-user/ebs/msmarco'


    #### Read the corpus file containing all the passages. Store them in the corpus dict
    corpus = {}  # dict in the format: passage_id -> passage. Stores all existing passages
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            corpus[pid] = passage

    ### Read the train queries, store in queries dict
    queries = {}  # dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv') #'queries.train.tsv'  'queries.train.qspladev2.top10.tsv'  'queries.train.spladedoc3_10.tsv'
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query


    ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')

    with gzip.open(ce_scores_file, 'rb') as fIn:
        ce_scores = pickle.load(fIn)

    # As training data we use hard-negatives that have been mined using various systems
    hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives-splade.jsonl.gz')


    train_queries = {}
    negs_to_use = None


    with gzip.open(hard_negatives_filepath, 'rt') as fIn:
        for line in tqdm(fIn):
            if max_passages > 0 and len(train_queries) >= max_passages:
                break
            data = json.loads(line)

            #Get the positive passage ids
            pos_pids = data['pos']

            if len(pos_pids) == 0:  #Skip entries without positives passages
                continue

            pos_min_ce_score = min([ce_scores[data['qid']][pid] for pid in data['pos']])
            ce_score_threshold = pos_min_ce_score - ce_score_margin

            neg_pids = []

            #Get the hard negatives

            if negs_to_use is None:
                negs_to_use = ['splade']

            for system_name in negs_to_use:
                if system_name not in data['neg']:
                    continue

                system_negs = data['neg'][system_name]
                negs_added = 0
                for pid in system_negs:
                    pid = int(pid)
                    if pid not in neg_pids:
                        neg_pids.append(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break

            if ((len(pos_pids) > 0 and len(neg_pids) > 0)):
                train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}

    for qid in train_queries:          
        pos_list = []
        for posid in train_queries[qid]['pos']:
            if posid in train_queries[qid]['neg']:
                pos_list.append([train_queries[qid]['neg'].index(posid) + 1, posid])
            else:
                pos_list.append([len(train_queries[qid]['neg']) + 1, posid])

        target_scores = [[pid[1], ce_scores[qid][pid[1]]] for pid in pos_list] + [[pid, ce_scores[qid][pid]] for pid in train_queries[qid]['neg']]
        target_scores = sorted(target_scores, key = lambda x: -x[1])
        target_ids = [x[0] for x in target_scores]

        train_queries[qid]['pos'] = [x + [target_ids.index(x[1]) + 1] for x in pos_list]
        train_queries[qid]['neg'] = [[x[0] + 1, x[1], target_ids.index(x[1]) + 1] for x in enumerate(train_queries[qid]['neg'])]
        train_queries[qid]['splade_pos'] = [x + [target_ids.index(x[1]) + 1] for x in pos_list]
        train_queries[qid]['splade_neg'] = [x for x in train_queries[qid]['neg']]


    evaluate_trainining(word_embedding_model.to('cuda'), word_embedding_model.tokenizer, train_queries, corpus, 60, 60)


    #open(os.path.join(model_save_path, "train_queries.json"), "w").write(json.dumps(train_queries))