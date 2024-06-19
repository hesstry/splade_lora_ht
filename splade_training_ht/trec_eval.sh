trec_binary=/expanse/lustre/projects/csb185/yifanq/trec_eval
trec_file=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/evaluate/run-files/index_msmarco_dev.trec
qrels_file=/expanse/lustre/projects/csb185/yifanq/qrels.dev.tsv

# # NDCG
# ./trec_eval -m recall qrels.dev.tsv xx.trec
$trec_binary -m recall $qrels_file $trec_file

# # MRR@10
# ./trec_eval -m recip_rank -M 10 qrels.dev.tsv xx.trec
$trec_binary -m recip_rank -M 10 $qrels_file $trec_file