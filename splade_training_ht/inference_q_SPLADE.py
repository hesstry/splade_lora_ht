import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
from tqdm import tqdm
import gzip 
import json
import os

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


agg = "max"
model_type_or_dir = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer

model = Splade(model_type_or_dir, agg=agg)
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

scale = 50
i = 0

data_file = '/expanse/lustre/projects/csb185/thess/splade/data/msmarco/dev_queries/raw.tsv'
#queries_filepath = os.path.join(data_folder, 'queries.dev.tsv')
with open(data_file) as f, open(os.path.join(sys.argv[2], "queries.dev.tsv"), "w") as fo:
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
