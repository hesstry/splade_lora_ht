import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
from tqdm import tqdm
import gzip 
import json
import os

from models import *

from sbert import *

class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir, agg="max", thresholding="qd"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max", "avg")
        self.agg = agg
        self.thresholding = thresholding
    
    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        if self.agg == "max":
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive
        elif self.agg == "sum":
            return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        else:
            length = torch.sum(kwargs["attention_mask"].unsqueeze(-1))
            sum_val = torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return sum_val / length

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
input_type = "d"

# the model is loaded using max pooling automatically
model = SpladeThresholding(model_name_or_path=model_type_or_dir, max_seq_length=max_seq_length, thresholding=thresholding, is_training=is_training, input_type=input_type)
model.load_state_dict(torch.load(model_state_dict))

# The below was a sanity check to ensure the model isn't starting from an initial point of finetuning where these new parameters are 0
# assert model.q_thres != 0, f"q_thres: {model.q_thres}"
# assert model.d_thres != 0, f"d_thres: {model.d_thres}"
# assert model.q_mean_thres != 0, f"q_mean_thres: {model.q_mean_thres}"
# assert model.d_mean_thres != 0, f"d_mean_thres: {model.d_mean_thres}"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

print("MODEL LOADED :D :D :D")
print(f"INSTANTIATED MODEL WITH FOLLOWING ATTRIBUTES:")
print(f"THRESHOLDING: {model.thresholding}")
print(f"INFERENCE: {model.is_training == False}")
print(f"ENCODING ONLY DOCUMENTS: {model.input_type == 'd'}")

# exit()

scale = 100
file_per = 100000
i = 0
starting_i = int(sys.argv[5])
ending_i = int(sys.argv[6])

# the below are hard-coded and this will probably need to change for the final version of this pipeline
# data_folder = '/expanse/lustre/projects/csb176/yifanq/msmarco_yingrui/'
# collection_filepath = os.path.join(data_folder, 'collection.tsv')

collection_filepath=sys.argv[7]

fo = None
with open(collection_filepath) as f:
    print(f"ENCODING CORPUS FROM {starting_i} TO {ending_i} :D :D :D")
    for line in tqdm(f):
        if i == ending_i:
            break
        if i < starting_i:
            i += 1
            continue
        if i % file_per == 0:
            if fo is not None:
                fo.close()
            fo = gzip.open(os.path.join(out_dir, f"file_{i // file_per}.jsonl.gz"), "w")

        did, doc = line.strip().split("\t")     
        with torch.no_grad():
            tokenized = tokenizer(doc, return_tensors="pt", truncation=True).to('cuda')
            # print(tokenized)
            doc_rep = model(tokenized).squeeze()  # (sparse) doc rep in voc space, shape (30522,)
            # print()
            # doc_rep = model.encode([doc])
            # print(doc_rep.shape)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        #print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        # adjust for scaling here, quantization purposes, no need to worry about proper thresholding as the model handles this in its forward function
        d = {reverse_voc[k]: int(v * scale) for k, v in zip(col, weights)}
        outline = json.dumps({"id": int(did), "content": doc, "vector": d}) + "\n"
        fo.write(outline.encode('utf-8'))
        fo.flush()
        i += 1
print("FINISHED :D :D :D")
fo.close()