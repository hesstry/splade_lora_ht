This is meant to instruct one on how to use the following pipeline which relies upon sentence transformers in order to load and train a SPLADE model. 

* The main changes are simply allowing for specifying which thresholding technique to use, and allowing for proper checkpointing so one needs not manually input threshold parameters for inference
    * This introduces a few new arguments:
    * ```--thresholding```
        * ```qd``` implies using a single learnable threshold for queries and documents respectively, tested
        * ```plus_mean``` implies using the above threshold technique plus a mean thresholding for query and document embeddings means, not tested
        * ```mean``` implies using only mean thresholding from query and document embeddings, not tested
            * early results for mean thresholding show this has little to no regularizing effects due to sparse vectors with means approaching 0 due to large sizes with few non-zero entries
        * more TBD
    * ```--checkpoint``` denoting whether or not a checkpoint using a ```saved state_dict.pt``` file is desired
    * ```--state_dict_path``` which is self-explanatory and is only used if ```--checkpoint``` is provided
        * **NOTE**: A proper initial checkpoint must be provided to instantiate the model class, and the ```state_dict_path``` is used to correctly populate all model parameters

### Notes:
* Don't forget to supply a proper ```--save_path``` when running the python script :sweat_smile:

### Some important changes to come:
* :white_check_mark: Allow for checkpointing 
* :white_check_mark: Allow model to handle all thresholding in forward function for simplified inference pipeline
    * Scaling is manually done in the inference scripts for queries and documents
* TBD

### End-to-end pipeline for fine-tuning, embedding, and inference:

#### Fine-tuning
* Using the ```run_train_orig.sh``` script with a proper model checkpoint (included in the script for now until proper organization occurs)
* The above script relies upon ```train_splade.py```
    * Multi-GPU should be possible using torchrun, but this hasn't been tested
    * Training takes roughly 40hours for 100,000 iterations (around the time when, empirically, the model converges)
    * **PLEASE NOTE**: Training steps are multiplied by gradient accumulation steps as sentence transformer's fit function treats a single gradient calculation as a single step, but we want a single step to be one weight update, hence
        * This changes the training arguments slightly: If desired train ```steps = n```, and gradient ```accumulation steps = a```, then train steps argument becomes: ```na```
        * This is accounted for in the training script automatically, but should be noted since this is weird behavior... to me at least.
* example command to train for 100000 steps with a gradient accumulation of 4: ```/bin/bash -c "python -m train_splade --model_name $model_path --train_batch_size 32 --accum_iter 4 --epochs 100000 --warmup_steps 6000 --loss_type marginmse --continues --num_negs_per_system 20 --training_queries $train_queries --thresholding qd"```

#### Checkpointing
* Note that one may now faithfully checkpoint the model and began training with the last learned threshold parameters, this is accounted for by saving a ```state_dict.pt``` file, whose path should be provided as an argument if one wishes to begin finetuning from a checkpoint
    * This is accounted for in the ```sbert.py``` script, which is used by the sentence transformers API
* **NOTE**: To do so one must supply the ```--checkpoint``` argument, and a proper ```--state_dict_path``` argument so the threshold parameters can be properly instantiated from this checkpoint
* The script will automatically save a ```state_dict.pt``` file which is the state dict file one should provide when trying to reload this model from a checkpoint

### Inference
**One can use the same expanse-slurm environment setup for all of the following steps, an example environment can be found in embed_docs.sh**

**NOTES**: 
* Scaling: This is accounted for during inference, such that the final outputs will have already been scaled, one can find this is done in the ```inference_SPLADE.py``` and the ```inference_q_SPLADE.py``` scripts
* ...
#### Uncompressed embeddings --- index
Example bash script that can be used for the expanse slurm system with proper commands:
```
#!/bin/bash
#SBATCH -A $EXPANSE_PROJECT
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUTPUT_LOG_PATH
#SBATCH --error=$ERROR_LOG_PATH
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

conda activate $CONDA_ENV

# MODEL CHECKPOING INFORMATION
# This is the model path in which to instantiate the class, ensure it is of the same type as the original path used for finetuning
# eg: /path/to/model/model_directory
MODEL_CHECKPOINT=?

# STATE_DICT_PATH INFORMATION
# The checkpoint should be a state_dict.pt file, such that the instantiated model will inherit the proper learned thresholding parameters
# eg: /path/to/state/dict/file.pt
STATE_DICT_PATH=?

# C_EMBS_OUTPUT INFORMATION
# This variable denotes where to store the embedded documents
# eg: /path/to/store/embeddings
C_EMBS_OUTPUT=?

# THRESHOLDING INFORMATION
# This variable denotes how inference should be done, this accounts for using the proper thresholds depending on which technique was used
THRESHOLDING=? # from [qd, mean, plus_mean], naming is a work in progress

# COLLECTION FILEPATH INFORMATION
# e.g. /path/to/msmarco/corpus
COLLECTION_FILEPATH=?

# ENCODING DOCUMENTS INFORMATION
# Run three separate SBATCH commands, one for each subgroup of the corpus, uncomment the desired subsample to encode
# eg: MSMARCO has around 8.8mil, so can overestimate and embedding script handles rest
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $STATE_DICT_PATH $C_EMBS_OUTPUT $THRESHOLDING 0 3000000 $COLLECTION_FILEPATH"
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $STATE_DICT_PATH $C_EMBS_OUTPUT $THRESHOLDING 3000000 6000000 $COLLECTION_FILEPATH"
# /bin/bash -c "python inference_SPLADE.py $MODEL_CHECKPOINT $STATE_DICT_PATH $C_EMBS_OUTPUT $THRESHOLDING 6000000 9000000 $COLLECTION_FILEPATH"
```

One can make this better by forcing three SBATCH scripts to run at once with a for loop and proper refactoring, for now this method works fine

#### Uncompressed embeddings --- queries
* Using a similar script as the above with same expanse-slurm environment settings but now using the ```inference_q_SPLADE.py``` script
```
# encode queries
/bin/bash -c "python inference_q_SPLADE.py $MODEL_CHECKPOINT $STATE_DICT_PATH $Q_OUTPUT_DIR $THRESHOLDING $QUERIES_FILEPATH"
```

This will save a ```queries.dev.tsv``` file in a BOW-style representation

#### Build the index to prepare it for compression using PISA
This builds all of the proper files needed for PISA compression and evaluation, an example script can be found in ```build_index.sh```

```
# build index

# C_EMBS_OUTPUT from embed_docs.sh
C_EMBS_OUTPUT=?

# OUTPUT_PREFIX
OUTPUT_PREFIX=/path/to/file/prefix

# NUM_FILES number of files generated from the embed_docs.sh
NUM_FILES=?

/bin/bash -c "python buildIndex.py $C_EMBS_OUTPUT $OUTPUT_PREFIX $NUM_FILES"
```

The above will produce the following files:
* ```/path/to/file/prefix.freqs```
* ```/path/to/file/prefix.sizes```
* ```/path/to/file/prefix.id```

#### Process encoded queries to get query id files

An example script can be found in ```generate_queries.sh```

```
# process encoded queries and generate id listing file

# OUTPUT_PREFIX INFORMATION
# This is the same prefix provided in the build_index.sh script
OUTPUT_PREFIX=?

# ENCODED_QUERY_FILE INFORMATION
# This is the same file generated in the embed_queries.sh script
ENCODED_QUERY_FILE=?

# QUERY_ID_PATH INFORMATION
# This is the file created by running this script
QUERY_ID_PATH=?

python generate_queries.py $OUTPUT_PREFIX $ENCODED_QUERY_FILE $QUERY_ID_PATH
```

### Compress, index and search

```
# EXPANSE SLURM STUFF

collection=/path/to/index/prefix # from above steps
OUTPUT_QUERY_ID=/path/to/queries.id # from steps above
pisa_build=/path/to/pisa/binary
query=/path/to/evaluation/queries
msmarco_lex_path=/path/to/lexical/dataset

# compress index
# $pisa_build/build/bin/compress_inverted_index --encoding block_simdbp --collection $collection --output $collection.block_simdbp.idx

# create wand data 
# $pisa_build/build/bin/create_wand_data --collection $collection --output $collection.fixed-40.bmw --block-size 40 --scorer quantized

# calculate relevance (multiple thread)

# $pisa_build/build/bin/evaluate_queries --encoding block_simdbp --index $collection.block_simdbp.idx --wand $collection.fixed-40.bmw --documents $msmarco_lex_path --algorithm maxscore -k 1000 --queries $OUTPUT_QUERY_ID --scorer quantized --run "$collection-$query" > evaluate/run-files/index"_"$query".trec"

# evaluate time (single thread)

$pisa_build/build/bin/queries --encoding block_simdbp --index $collection.block_simdbp.idx --wand $collection.fixed-40.bmw --algorithm maxscore -k 1000 --queries $OUTPUT_QUERY_ID --scorer quantized
```

### Length analysis
TODO