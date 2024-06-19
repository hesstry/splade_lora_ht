
import os 

cwd = os.getcwd()
print(cwd)

import json
import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH

from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import asdict

from huggingface_hub import login

import accelerate
from peft import LoraConfig, get_peft_model, TaskType

# with torchrun, relies upon splade/splade directory, need to add parent splade directory to path
import sys
sys.path.append('/expanse/lustre/projects/csb185/thess/splade')

from splade.hf.trainers import IRTrainer
from splade.hf.collators import L2I_Collator
from splade.hf.datasets import L2I_Dataset, TRIPLET_Dataset
from splade.hf.models import  SPLADE, DPR, QLoRALLaMa
from splade.hf.convertl2i2hf import convert
from splade.utils.utils import get_initialize_config

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME,version_base="1.2")
def hf_train(exp_dict: DictConfig):

    # mapping yaml/hydra conf into HF data structure
    exp_dict, _, _, _ = get_initialize_config(exp_dict, train=True)
    model_args,data_args,training_args = convert(exp_dict)

    print(f"TRAINING ARGS: {training_args}")

    # exit()

    # need to accomodate lack of padding in llama tokenizer and changing eos_token
    # llama is a bit different, adds EOS token at end and makes CLS = EOS
    # adds PAD token since this is required with huggingface
    # this is good since resizing tokenizer can lead to unstable training so not needing to add special tokens is useful
    if "llama" in model_args.tokenizer_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, add_eos_token=True, use_fast=True)
        tokenizer.cls_token = tokenizer.eos_token # makes the cls </s> to mimic RepLLaMa
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    if "lora" in exp_dict["config"]:
        lora = exp_dict["config"]["lora"]
    else:
        lora = None

    if "qlora" in lora:
        qlora = lora["qlora"]
    else:
        qlora = None

    print("LOADING MODEL")

    # initialize the model: dense or a splade model (splade-doc,(a)symetric splade etc)
    if model_args.dense:
        print("PRINTING MODEL ARGS")
        print(model_args)
        access_token = str(exp_dict["hf"]["token"])
        login(token = access_token)

        if "llama" in model_args.tokenizer_name_or_path.lower():
            model = QLoRALLaMa(
                model_args.model_name_or_path,
                shared_weights=model_args.shared_weights,
                n_negatives=data_args.n_negatives,
                tokenizer=tokenizer, 
                model_q=model_args.model_q, 
                pooling=model_args.dense_pooling, 
                quantization_config_args=qlora,
                lora_config_args=lora,
                training=True
                )

        else:
            model = DPR(
                model_args.model_name_or_path,
                shared_weights=model_args.shared_weights,
                n_negatives=data_args.n_negatives,
                tokenizer=tokenizer, 
                model_q=model_args.model_q, 
                pooling=model_args.dense_pooling, 
                quantization_config=quantization_config
                )

    else:
        model = SPLADE(
            model_args.model_name_or_path,shared_weights=model_args.shared_weights,n_negatives=data_args.n_negatives,
            tokenizer=tokenizer, splade_doc=model_args.splade_doc, model_q=model_args.model_q)
            #adapter_name=model_args.adapter_name, adapter_config=model_args.adapter_config, load_adapter=model_args.load_adapter)

        if lora:
            print("LORA CONFIG DETAILS")
            lora_config = model.get_lora_config(model, lora)
            print(lora_config)
            model = get_peft_model(model, lora_config)

            model.print_trainable_parameters()

    print(model)
    subprocess.run("nvidia-smi")

    # load the dataset
    data_collator= L2I_Collator(tokenizer=tokenizer,max_length=model_args.max_length)
    if data_args.training_data_type == 'triplets':
         dataset = TRIPLET_Dataset(data_dir=data_args.training_data_path)
    else:
        dataset = L2I_Dataset(training_data_type=data_args.training_data_type, # training file type
                              training_file_path=data_args.training_data_path, # path to training file
                              document_dir=data_args.document_dir,             # path to document file (collection)
                              query_dir=data_args.query_dir,                   # path to queri=y file
                              qrels_path=data_args.qrels_path,                 # path to qrels
                              n_negatives=data_args.n_negatives,               # nb negatives in batch
                              nqueries=data_args.n_queries,                    # consider only a subset of <nqueries> queries
                              )

    
    trainer = IRTrainer(model=model,                         # the instantiated 🤗 Transformers model to be trained
                        args=training_args,                  # training arguments, defined above
                        train_dataset=dataset,
                        data_collator=data_collator.torch_call,
                        tokenizer=tokenizer,
                        shared_weights=model_args.shared_weights,  # query and document model shared or not
                        splade_doc=model_args.splade_doc,          # model is a spladedoc model
                        n_negatives=data_args.n_negatives,         # nb negatives in batch 
                        dense=model_args.dense,                    # is the model dense or not (DPR or SPLADE)
                        llama=True if "llama" in model_args.tokenizer_name_or_path.lower() else False)
    
    last_checkpoint = None
    if training_args.resume_from_checkpoint: #os.path.isdir(training_args.output_dir) and  not training_args.overwrite_output_dir:
        last_checkpoint  =  get_last_checkpoint(training_args.output_dir)

    if  trainer.is_world_process_zero():
        print(OmegaConf.to_yaml(exp_dict))


    subprocess.run("nvidia-smi")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    final_path = os.path.join(training_args.output_dir,"model")
    os.makedirs(final_path,exist_ok=True)

    # need to merge them to save them as if one complete model, merging only works if not quantized
    if lora["use"] and not qlora:
        model = model.merge_and_unload()

    # by simply saving the model we are also saving the adapters, and loading then has an extra line where we load the adapters after the model
    # and then wrap this in PeftModel.from_pretrained(model, adapter_path)
    # advice from: https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997

    trainer.save_model(final_path)

    #trainer.create_model_card()   # need .config
    
    if  trainer.is_world_process_zero():
        with open(os.path.join(final_path, "model_args.json"), "w") as write_file:
            json.dump(asdict(model_args), write_file, indent=4)
        with open(os.path.join(final_path, "data_args.json"), "w") as write_file:
            json.dump(asdict(data_args), write_file, indent=4)
        with open(os.path.join(final_path, "training_args.json"), "w") as write_file:
            json.dump(training_args.to_dict(), write_file, indent=4)


if __name__ == "__main__":
    hf_train()