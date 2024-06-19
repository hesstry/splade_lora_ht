#!/bin/bash
#SBATCH -A csb185
#SBATCH --job-name="evaluate_time"
#SBATCH --output="../slurm_logs/mean_thresholding/evaluate_time.%j.%N.out"
#SBATCH --error="../slurm_logs/mean_thresholding/evaluate_time.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH -t 10:00:00

collection=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/collection/index
OUTPUT_QUERY_ID=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/queries/queries.id
pisa_build=/expanse/lustre/projects/csb176/yifanq/pisa
query=msmarco_dev
msmarco_lex_path=/expanse/lustre/projects/csb185/yifanq/msmarco.lex

# compress index
# $pisa_build/build/bin/compress_inverted_index --encoding block_simdbp --collection $collection --output $collection.block_simdbp.idx

# create wand data 
# $pisa_build/build/bin/create_wand_data --collection $collection --output $collection.fixed-40.bmw --block-size 40 --scorer quantized

# # calculate relevance (multiple thread)

# $pisa_build/build/bin/evaluate_queries --encoding block_simdbp --index $collection.block_simdbp.idx --wand $collection.fixed-40.bmw --documents $msmarco_lex_path --algorithm maxscore -k 1000 --queries $OUTPUT_QUERY_ID --scorer quantized --run "$collection-$query" > evaluate/run-files/index"_"$query".trec"

# # evaluate time (single thread)

$pisa_build/build/bin/queries --encoding block_simdbp --index $collection.block_simdbp.idx --wand $collection.fixed-40.bmw --algorithm maxscore -k 1000 --queries $OUTPUT_QUERY_ID --scorer quantized