
from collections import defaultdict
run = defaultdict(dict)
with open("../../msmarco/wentai_splade_dev_top1000.trec") as f:
    for line in f:
        qid, _, did, rank, _, _ = line.strip().split("\t")
        if int(rank) <= 100:
            run[qid][did] = 1

with open("../../msmarco/wentai_splade_dev_top1000.tsv") as f, open("../../msmarco/wentai_splade_dev_top100.tsv", "w") as fo:
    for line in f:
        qid, did, _, _ = line.strip().split("\t")
        if  did in run[qid] and run[qid][did]== 1:
            fo.write(line)