#!/bin/bash
#SBATCH -A csb185
#SBATCH --job-name="testing_embed_docs"
#SBATCH --output="./output/testing_model_refactoring/testing_embed_docs.%j.%N.out"
#SBATCH --error="./output/testing_model_refactoring/testing_embed_docs.%j.%N.err"
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

# MODEL CHECKPOING INFORMATION
# This is the model path in which to instantiate the class, ensure it is of the same type as the original path used for finetuning
# eg: /path/to/model/model_directory
MODEL_CHECKPOINT=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/0_SpladeThresholding

# STATE_DICT_PATH INFORMATION
# The checkpoint should be a state_dict.pt file, such that the instantiated model will inherit the proper learned thresholding parameters
# eg: /path/to/state/dict/file.pt
STATE_DICT_PATH=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/state_dict.pt

# C_EMBS_OUTPUT INFORMATION
# This variable denotes where to store the embedded documents
# eg: /path/to/store/embeddings
C_EMBS_OUTPUT=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/doc_embs

# THRESHOLDING INFORMATION
# This variable denotes how inference should be done, this accounts for using the proper thresholds depending on which technique was used
THRESHOLDING=plus_mean

# COLLECTION FILEPATH INFORMATION
# Path to collection to embed e.g. MSMARCO corpus
COLLECTION_FILEPATH=/expanse/lustre/projects/csb176/yifanq/msmarco_yingrui/collection.tsv

# STARTING ID TO ENDING ID
# STARTING_ID=0
# ENDING_ID=3000000

# STARTING_ID=3000000
# ENDING_ID=6000000

STARTING_ID=6000000
ENDING_ID=9000000

# ENCODING DOCUMENTS INFORMATION
# Run three separate SBATCH commands, one for each subgroup of the corpus
# eg: MSMARCO has around 8.8mil, so can overestimate and embedding script handles rest
/bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $STATE_DICT_PATH $C_EMBS_OUTPUT $THRESHOLDING $STARTING_ID $ENDING_ID $COLLECTION_FILEPATH"
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $STATE_DICT_PATH $C_EMBS_OUTPUT $THRESHOLDING 3000000 6000000 $COLLECTION_FILEPATH"
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $STATE_DICT_PATH $C_EMBS_OUTPUT $THRESHOLDING 6000000 9000000 $COLLECTION_FILEPATH"