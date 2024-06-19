#!/bin/bash
#SBATCH -A csb185
#SBATCH --job-name="only_mean_test_proper_run"
#SBATCH --output="./output/only_mean/only_mean_test_proper_run.%j.%N.out"
#SBATCH --error="./output/only_mean/only_mean_test_proper_run.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH -t 10:00:00

module purge
module load gpu
module load slurm
module load anaconda3/2021.05

# load conda env variables
__conda_setup="$('/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh" ]; then
        . "/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh"
    else
        export PATH="/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate splade_env

# MODEL_CHECKPOINT=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/splade_distill_num1_marginmse_mrr_diff_denoiseFalse_num20_marginmse1-lambda0.01-0.008_lr2e-05-batch_size_32x4-2024-06-03-relu-q1l-dflop-1sqhd-thess/75000/0_MLMTransformer
# C_OUTPUT_DIR=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/collection
# Q_OUTPUT_DIR=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/queries


# NOTES: Checkpoints don't work for this pipeline, hence below is simply a model trained for 10000 steps
# RUN WITH ONLY_MEAN
# MODEL_CHECKPOINT=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/only_mean_90000_100000/10000/0_MLMTransformer
# C_OUTPUT_DIR=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/index
# C_OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/index/index
# Q_OUTPUT_DIR=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/queries
# Q_OUTPUT_ID=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/queries/queries.id

# ONLY MEAN WITH 45,000 STEPS
MODEL_CHECKPOINT=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/only_mean_45000/45000/0_MLMTransformer
C_OUTPUT_DIR=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/index
C_OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/index/index
Q_OUTPUT_DIR=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/queries
Q_OUTPUT_ID=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/only_mean/queries/queries.id

# encode documents
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $C_OUTPUT_DIR 0 3000000"
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $C_OUTPUT_DIR 3000000 6000000"
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $C_OUTPUT_DIR 6000000 9000000"

# # encode queries
# /bin/bash -c "python inference_q_SPLADE.py $MODEL_CHECKPOINT $Q_OUTPUT_DIR"

# build index
# /bin/bash -c "python buildIndex.py  $JSON_PATH_PREFIX $OUTPUT_PREFIX $SCALE $NUM_JSON_FILES $DOC_THRESH $THRESH_TYPE"
C_OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/collection/index
SCALE=100
NUM_JSON_FILES=89
# SCALING ACCOUNTED FOR IN buildIndex.py, can provide float here
DOC_THRESH="0.0"
MEAN_THRESH="0.6751"
THRESH_TYPE="only_mean"
/bin/bash -c "python buildIndex.py $C_OUTPUT_DIR $C_OUTPUT_PREFIX $SCALE $NUM_JSON_FILES $DOC_THRESH $MEAN_THRESH $THRESH_TYPE"

# # process encoded queries
# python generate_queries.py $OUTPUT_PREFIX $ENCODED_QUERY_FILE $OUTPUT_QUERY_ID
# ENCODED_QUERY_FILE=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/queries/queries.dev.tsv
# OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/collection/index
# OUTPUT_QUERY_ID=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/queries/queries.id
# # TODO: NEEDS TO BE x50
# Q_THRESH="0.6571"
# python generate_queries.py $OUTPUT_PREFIX $ENCODED_QUERY_FILE $OUTPUT_QUERY_ID $Q_THRESH $THRESH_TYPE

# NOTES: Accidentally did mean only thresholding for this checkpoint, so I now have a mean thresholding one with 45000 steps
# TODO RUN WITH PLUS_MEAN
# MODEL_CHECKPOINT=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/plus_mean/plus_mean_55000_100000/45000/0_MLMTransformer
# C_OUTPUT_DIR=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/plus_mean/index
# C_OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/plus_mean/index/index
# Q_OUTPUT_DIR=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/plus_mean/queries
# Q_OUTPUT_ID=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/plus_mean/queries/queries.id

# encode documents 
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $C_OUTPUT_DIR 0 3000000"
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $C_OUTPUT_DIR 3000000 6000000"
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $C_OUTPUT_DIR 6000000 9000000"

# # encode queries
# /bin/bash -c "python inference_q_SPLADE.py $MODEL_CHECKPOINT $Q_OUTPUT_DIR"

# build index
# /bin/bash -c "python buildIndex.py  $JSON_PATH_PREFIX $OUTPUT_PREFIX $SCALE $NUM_JSON_FILES $DOC_THRESH"
# C_OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/collection/index
# SCALE=100
# NUM_JSON_FILES=89
# SCALING ACCOUNTED FOR IN buildIndex.py, can provide float here
# DOC_THRESH="0.5910"
# MEAN_THRESH="0.123"
# THRESH_TYPE="plus_mean"
# /bin/bash -c "python buildIndex.py $C_OUTPUT_DIR $C_OUTPUT_PREFIX $SCALE $NUM_JSON_FILES $DOC_THRESH $MEAN_THRESH $THRESH_TYPE"

# # process encoded queries
# python generate_queries.py $OUTPUT_PREFIX $ENCODED_QUERY_FILE $OUTPUT_QUERY_ID
# ENCODED_QUERY_FILE=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/queries/queries.dev.tsv
# OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/collection/index
# OUTPUT_QUERY_ID=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/index/queries/queries.id
# # TODO: NEEDS TO BE x50
# Q_THRESH="0.5116"
# python generate_queries.py $OUTPUT_PREFIX $ENCODED_QUERY_FILE $OUTPUT_QUERY_ID $Q_THRESH $THRESH_TYPE