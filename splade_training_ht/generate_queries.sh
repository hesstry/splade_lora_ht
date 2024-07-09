#!/bin/bash
#SBATCH -A csb185
#SBATCH --job-name="testing_build_queries"
#SBATCH --output="./output/testing_model_refactoring/testing_build_queries.%j.%N.out"
#SBATCH --error="./output/testing_model_refactoring/testing_build_queries.%j.%N.err"
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

# process encoded queries and generate id listing file

# OUTPUT_PREFIX INFORMATION
# This is the same prefix provided in the build_index.sh script
OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/doc_embs/index

# ENCODED_QUERY_FILE INFORMATION
# This is the same file generated in the embed_queries.sh script
ENCODED_QUERY_FILE=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/queries/queries.dev.tsv

# QUERY_ID_PATH INFORMATION
# This is the file created by running this script
QUERY_ID_PATH=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/queries/queries.id

python generate_queries.py $OUTPUT_PREFIX $ENCODED_QUERY_FILE $QUERY_ID_PATH