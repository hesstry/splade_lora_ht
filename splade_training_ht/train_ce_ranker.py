#FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
#Original License Apache2, NOTE: Trained MSMARCO models are NonCommercial (from dataset License)

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
import tqdm
from torch.utils.data import Dataset
import random
from shutil import copyfile
import pickle
import argparse
import losses
import torch
from collections import defaultdict
from data import *
from evaluate_training import evaluate_trainining
import transformers
#from colbert_model import DETeacher

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used ? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--accum_iter", default=1,type=int)
parser.add_argument("--loss_type", default="marginmse",type=str)
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--denoise", default=False, action="store_true")
parser.add_argument("--deteacher", default=False, action="store_true")
parser.add_argument("--continues", default=False, action="store_true")
parser.add_argument("--nway", default=1, type=int)
parser.add_argument("--prf", default=False, action="store_true")


args = parser.parse_args()

logging.info(str(args))

train_batch_size = args.train_batch_size  # Increasing the train batch size generally improves the model performance, but requires more GPU memory
model_name = args.model_name
max_passages = args.max_passages
ce_score_margin = args.ce_score_margin
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it implies more GPU memory needed
num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train

# Load our embedding model
logging.info("Create new SBERT model")
word_embedding_model = models.CETransformerSeq(model_name, max_seq_length=max_seq_length)
print(len(word_embedding_model.tokenizer))
tokens = ["[unused0]", "[unused1]", "[unused2]"] #[unused0] for query, [unused1] for doc, [unused2] for query expansion
word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
print(len(word_embedding_model.tokenizer))

if args.continues:
    checkpoint = torch.load(os.path.join(model_name, "checkpoint.pt"), map_location='cpu')
    word_embedding_model.load_state_dict(checkpoint)

model = SentenceTransformerA(modules=[word_embedding_model])
model_save_path = f'output/ce_num{args.num_negs_per_system}_marginkldiv{args.nway}-batch_size_{train_batch_size}x{args.accum_iter}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

# Write self to path
os.makedirs(model_save_path, exist_ok=True)

train_script_path = os.path.join(model_save_path, 'train_script.py')
copyfile(__file__, train_script_path)
with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

### Now we read the MS MARCO dataset

data_folder = '../../msmarco'


#### Read the corpus file containing all the passages. Store them in the corpus dict
corpus = {}  # dict in the format: passage_id -> passage. Stores all existing passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage

### Read the train queries, store in queries dict
'''
queries = {}  # dict in the format: query_id -> query. Stores all training queries
if args.prf:
    queries_filepath = os.path.join(data_folder, 'queries.train.spladedoc3_10.tsv') #'queries.train.tsv'  'queries.train.qspladev2.top10.tsv'  'queries.train.spladedoc3_10.tsv'
else:
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query


'''
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
if not os.path.exists(ce_scores_file):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

train_queries = dict()

with open("output/colbert_dynamic_fromwarmup_num20_marginkldiv_position5-batch_size_8x4-2022-07-18_06-05-04/num1/train_queries.json") as f:
    for line in f:
        train_queries[line.split("\t")[0]] = json.loads(line.split("\t")[1])
        train_queries[line.split("\t")[0]]['neg'] = train_queries[line.split("\t")[0]]['neg'][:args.num_negs_per_system]

#train_loss = losses.MultipleNegativesRankingLossCE(model=model)
train_loss = losses.MarginKLDivLossCE(model=model, scaled=False)


train_dataset = MSMARCODatasetCE(queries=train_queries, ce_scores = ce_scores, corpus=corpus, num_neg=args.nway, prf=args.prf, topk = 20, loss_type = "marginkldiv_position")
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
logging.info("Start model fitting...")

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=10000,
          optimizer_params = {'lr': args.lr},
          accum_iter = args.accum_iter)
          #schedulers = [scheduler],
          #optimizers = [optimizer])


