import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
from tqdm import tqdm
import gzip 
import json
import os

from models import *

class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg
    
    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        if self.agg == "max":
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive
        else:
            return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)


# agg = "max"
# model_type_or_dir = sys.argv[1]
# out_dir = sys.argv[2]
# model = Splade(model_type_or_dir, agg=agg)

# max pooling is already applied in the base model so this parameter is no longer needed
# agg = "max"
model_type_or_dir = sys.argv[1]

# ensure this bad boy is precisely the model checkpoint you wish to use
model_state_dict = sys.argv[2]

# this will be the directory that stores all of the "file_i.jsonl.gz" files
out_dir = sys.argv[3]

# this should be the same as the finetuning
max_seq_length = 256

# this denotes which thresholding type to use, the naming of these thresholding types is a work in progress [qd, mean, plus_mean]
thresholding = sys.argv[4]

# this allows the model to use the proper inference code block in its forward function
is_training = False

# providing the input-type allows the model to know which code block to execute during inference, this differs between queries and documents when thresholding
input_type = "q"

# the model is loaded using max pooling automatically
model = SpladeThresholding(model_name_or_path=model_type_or_dir, max_seq_length=max_seq_length, thresholding=thresholding, is_training=is_training, input_type=input_type)
model.load_state_dict(torch.load(model_state_dict))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = 50
i = 0

queries_file = sys.argv[5]
#queries_filepath = os.path.join(data_folder, 'queries.dev.tsv')
with open(queries_file) as f, open(os.path.join(out_dir, "queries.dev.tsv"), "w") as fo:
    for line in tqdm(f):
        did, doc = line.strip().split("\t")
        with torch.no_grad():
            tokenized = tokenizer(doc, return_tensors="pt", truncation=True).to('cuda')
            doc_rep = model(tokenized).squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        #print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
        q = []
        for tok in d:
            for _ in range(d[tok]):
                q.append(tok)
        #outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
        fo.write(f"{did}\t{' '.join(q)}\n")
        fo.flush()
        i += 1

'''
scale = 50
i = 0

with open("../msmarco_yingrui/queries.2020.tsv") as f, open(os.path.join(sys.argv[2], "queries.20.tsv"), "w") as fo:
    for line in tqdm(f):
        did, doc = line.strip().split("\t")
        with torch.no_grad():
            doc_rep = model(**tokenizer(doc, return_tensors="pt").to('cuda')).squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        #print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
        q = []
        for tok in d:
            for _ in range(d[tok]):
                q.append(tok)
        #outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
        fo.write(f"{did}\t{' '.join(q)}\n")
        fo.flush()
        i += 1

scale = 50
i = 0

with open("queries.train.5000.tsv") as f, open(os.path.join(sys.argv[2], "queries.train.tsv"), "w") as fo:
    for line in tqdm(f):
        did, doc = line.strip().split("\t")
        with torch.no_grad():
            doc_rep = model(**tokenizer(doc, return_tensors="pt").to('cuda')).squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        #print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
        q = []
        for tok in d:
            for _ in range(d[tok]):
                q.append(tok)
        #outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
        fo.write(f"{did}\t{' '.join(q)}\n")
        fo.flush()
        i += 1

scale = 50
i = 0

'''

# with open("../msmarco_yingrui/queries.dev.labeled.tsv") as f, open(os.path.join(sys.argv[2], "queries.dev.processed.tsv"), "w") as fo:
#     for line in tqdm(f):
#         did, doc = line.strip().split("\t")
#         with torch.no_grad():
#             doc_rep = model(**tokenizer(doc, return_tensors="pt").to('cuda')).squeeze()  # (sparse) doc rep in voc space, shape (30522,)

#         # get the number of non-zero dimensions in the rep:
#         col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
#         #print("number of actual dimensions: ", len(col))

#         # now let's inspect the bow representation:
#         weights = doc_rep[col].cpu().tolist()
#         d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
#         q = []
#         for tok in d:
#             for _ in range(d[tok]):
#                 q.append(tok)
#         #outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
#         fo.write(f"{did}\t{' '.join(q)}\n")
#         fo.flush()
#         i += 1
