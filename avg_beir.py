import json
import os
import numpy as np
import sys

rs = [1, 4, 8, 16, 32, 64]

exp_names = [f"lora_splade_ensemble_distil_monogpu_r{i}_a{i}" for i in rs]

exp_dirs_beir = [f"/expanse/lustre/projects/csb185/thess/splade/experiments/{i}/out/beir" for i in exp_names]

exp_dirs_trec19 = [f"/expanse/lustre/projects/csb185/thess/splade/experiments/{i}/out/TREC_DL_2019/perf.json" for i in exp_names]

exp_dirs_trec20 = [f"/expanse/lustre/projects/csb185/thess/splade/experiments/{i}/out/TREC_DL_2019/perf.json" for i in exp_names]

datasets = ["arguana", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa", "nfcorpus", "nq", "quora", "scidocs", "scifact", "trec-covid", "webis-touche2020"]

exp_results_beir = []
exp_results_trec19 = []
exp_results_trec20 = []

for i in range(len(rs)):
    exp_name = exp_names[i]
    exp_dir_beir = exp_dirs_beir[i]
    exp_dir_trec19 = exp_dirs_trec19[i]
    exp_dir_trec20 = exp_dirs_trec20[i]

    with open(exp_dir_trec19, "r") as fIn:
        results = json.load(fIn)
        exp_results_trec19.append(results["NDCG@10"])

    with open(exp_dir_trec20, "r") as fIn:    
        exp_results_trec20.append(results["NDCG@10"])

    ndcg_results = []
    for dataset in datasets:
        path = os.path.join(exp_dir_beir, dataset)
        path = os.path.join(path, "perf.json")

        with open(path, "r") as fIn:
            results = json.load(fIn)
            ndcg_results.append(results['NDCG@10'])

    print(f"FOR EXPERIMENT: {exp_name}")
    curr_avg = np.sum(ndcg_results)/len(ndcg_results)
    exp_results_beir.append(curr_avg)
    print(np.sum(ndcg_results)/len(ndcg_results))

print("EXP RESULTS BEIR AVGS")
print(exp_results_beir)

print("EXP RESULTS TREC19 NDCG@10")
print(exp_results_trec19)

print("EXP RESULTS TREC20 NDCG@10")
print(exp_results_trec20)