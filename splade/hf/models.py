import torch
from transformers import AutoModelForMaskedLM, AutoModel, BitsAndBytesConfig
from transformers.trainer import  logger
from transformers import PreTrainedModel
import os
from typing import Dict, List
from splade.utils.utils import generate_bow, clean_bow

import accelerate
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

class SpladeDoc(torch.nn.Module):

    def __init__(self, tokenizer,output_dim):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token = self.tokenizer.special_tokens_map["pad_token"]
        self.cls_token = self.tokenizer.special_tokens_map["cls_token"]
        self.sep_token = self.tokenizer.special_tokens_map["sep_token"]
        self.mask_token = self.tokenizer.special_tokens_map["mask_token"]
        self.pad_id = self.tokenizer.vocab[self.pad_token]
        self.cls_id = self.tokenizer.vocab[self.cls_token]
        self.sep_id = self.tokenizer.vocab[self.sep_token]
        self.mask_id = self.tokenizer.vocab[self.mask_token]
        self.output_dim = output_dim

    def forward(self, **tokens):
        q_bow = generate_bow(tokens["input_ids"], self.output_dim, device=tokens["input_ids"].device)
        q_bow = clean_bow(q_bow, pad_id = self.pad_id, cls_id=self.cls_id, sep_id=self.sep_id, mask_id=self.mask_id)
        return q_bow

    def _save(self, output_dir, state_dict=None):
        ## SAVE CHECKPOINT !
        pass    

class SPLADE(torch.nn.Module):
    
    @staticmethod
    def splade_max(output, attention_mask):
        # tokens: output of a huggingface tokenizer
        output = output.logits
        relu = torch.nn.ReLU(inplace=False)
        values, _ = torch.max(torch.log(1 + relu(output)) * attention_mask.unsqueeze(-1), dim=1)
        return values

    @staticmethod
    def passthrough(output, attention_mask):
        # tokens: output of a huggingface tokenizer
        return output


    def __init__(self, model_type_or_dir, tokenizer=None, shared_weights=True, n_negatives=-1, splade_doc=False, model_q=None, 
                 **kwargs):
        """
        output indicates which representation(s) to output ('MLM' for MLM model)
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None
        
        self.shared_weights = shared_weights       
        self.doc_encoder = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        
        self.output_dim=self.doc_encoder.config.vocab_size

        self.n_negatives = n_negatives
        self.splade_doc = splade_doc
        self.doc_activation = self.splade_max
        self.query_activation = self.splade_max if not self.splade_doc else self.passthrough

        if splade_doc:
            self.query_encoder = SpladeDoc(tokenizer=tokenizer,output_dim=self.doc_encoder.config.vocab_size)
        elif shared_weights:
            self.query_encoder = self.doc_encoder
        else:
            if model_q:
                self.query_encoder = AutoModelForMaskedLM.from_pretrained(model_q)
            else:
                self.query_encoder = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)

    def forward(self, **tokens):

        if not self.shared_weights or self.splade_doc:
            attention_mask = tokens["attention_mask"]
            input_ids = tokens["input_ids"] ##(bsz * (nb_neg+2) , seq_length)
            input_ids = input_ids.view(-1,self.n_negatives+2,input_ids.size(1)) ##(bsz, nb_neg+2 , seq_length)
            attention_mask = attention_mask.view(-1,self.n_negatives+2,attention_mask.size(1))
            docs_ids = input_ids[:,1:,:].reshape(-1,input_ids.size(2)) ##(bsz * (nb_neg+1) , seq_length)
            docs_attention = attention_mask[:,1:,:].reshape(-1,attention_mask.size(2))
            queries_ids = input_ids[:,:1,:].reshape(-1,input_ids.size(2))  ##(bsz * (1) , seq_length)
            queries_attention = attention_mask[:,:1,:].reshape(-1,attention_mask.size(2))

            queries_result = self.query_activation(self.query_encoder(input_ids=queries_ids,attention_mask=queries_attention), attention_mask=queries_attention)
            queries_result = queries_result.view(-1,1,queries_result.size(1))  ##(bsz, (1) , Vocab)
            docs_result = self.doc_activation(self.doc_encoder(input_ids=docs_ids,attention_mask=docs_attention),attention_mask=docs_attention)
            docs_result = docs_result.view(-1,self.n_negatives+1,docs_result.size(1))  ####(bsz, (nb_neg+1) , Vocab)
        else:
            representations = self.doc_activation(self.doc_encoder(**tokens),attention_mask=tokens["attention_mask"]) #TODO This should separate docs and queries and use their separate activations, for now is not a problem because they will always be the same if we are here.
            output = representations.view(-1,self.n_negatives+2,representations.size(1))
            queries_result = output[:,:1,:]
            docs_result = output[:,1:,:]
        return queries_result,docs_result

    def save(self,output_dir, tokenizer):
        # if self.doc_encoder_adapter_name and self.doc_encoder.active_adapters:
        #     self.doc_encoder.save_all_adapters(output_dir)
        # else:
        model_dict = self.doc_encoder.state_dict()
        torch.save(model_dict, os.path.join(output_dir,  "pytorch_model.bin"))
        self.doc_encoder.config.save_pretrained(output_dir)

        if not self.shared_weights:
            query_output_dir = os.path.join(output_dir,"query")
            os.makedirs(query_output_dir, exist_ok=True)
            # if self.doc_encoder_adapter_name and self.query_encoder.active_adapters:
                # self.query_encoder.save_all_adapters(query_output_dir)
            # else:
            self.query_encoder.save_pretrained(query_output_dir)
            self.query_encoder.config.save_pretrained(query_output_dir)
            if tokenizer:
                tokenizer.save_pretrained(query_output_dir)

        if tokenizer:
            tokenizer.save_pretrained(output_dir)

    def get_linear_modules(self, encoder):
        linear_modules = set()
        for i, j in encoder.named_modules():
            if "linear" in str(type(j)).lower():
                i = i.split(".")
                linear_modules.add(i[0] if len(i) == 1 else i[-1])

        return list(linear_modules)

    def get_lora_config(self, encoder, config_args):
        print("GETTING LORA CONFIG")
        linear_modules = self.get_linear_modules(encoder)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config_args["r"],
            lora_alpha=config_args["a"],
            # target_modules=["q_lin", "v_lin", "k_lin", "out_lin", "lin1", "lin2"],
            target_modules=linear_modules,
            lora_dropout=config_args["dropout"],
            bias=config_args["bias"],
            use_rslora=config_args["use_rslora"],
            # use_dora=config_args["use_dora"]
        )   

        return lora_config


class DPR(torch.nn.Module):

    def __init__(self, model_type_or_dir, shared_weights=True, n_negatives=-1, tokenizer=None, model_q=None, pooling='cls', quantization_config=None):
        """
        output indicates which representation(s) to output ('MLM' for MLM model)
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        self.shared_weights = shared_weights       
        self.doc_encoder = AutoModel.from_pretrained(model_type_or_dir, quantization_config=quantization_config, device_map="auto")
        self.n_negatives = n_negatives
        self.tokenizer = tokenizer
        self.pooling = pooling
        if shared_weights:
            self.query_encoder = self.doc_encoder
        else:
            if model_q:
                self.query_encoder = AutoModel.from_pretrained(model_q, quantization_config)
            else:
                self.query_encoder = AutoModel.from_pretrained(model_type_or_dir, quantization_config)

    @staticmethod
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def forward(self, **tokens):
        if not self.shared_weights:
            attention_mask = tokens["attention_mask"]
            input_ids = tokens["input_ids"]
            input_ids = input_ids.view(-1,self.n_negatives+2,input_ids.size(1))
            attention_mask = attention_mask.view(-1,self.n_negatives+2,attention_mask.size(1))
            docs_ids = input_ids[:,1:,:].reshape(-1,input_ids.size(2))
            docs_attention = attention_mask[:,1:,:].reshape(-1,attention_mask.size(2))
            queries_ids = input_ids[:,:1,:].reshape(-1,input_ids.size(2))
            queries_attention = attention_mask[:,:1,:].reshape(-1,attention_mask.size(2))

            query_result = self.query_encoder(input_ids=queries_ids,attention_mask=queries_attention)
            query_result = query_result[0]
            if self.pooling == 'mean':
                queries_result = self.mean_pooling(query_result, queries_attention)
            elif  self.pooling == 'cls': 
                queries_result = query_result[:,0,:]

            queries_result = queries_result.view(-1,1,queries_result.size(1))

            docs_result = self.doc_encoder(input_ids=docs_ids,attention_mask=docs_attention)[0]
            if self.pooling == 'mean':
                docs_result = self.mean_pooling(docs_result, queries_attention)
            else:
                docs_result = docs_result[:,0,:]
            docs_result = docs_result.view(-1,self.n_negatives+1,docs_result.size(1))
        else:
            output = self.doc_encoder(**tokens)[0]
            if self.pooling == 'mean':
                output = self.mean_pooling(output, tokens["attention_mask"])
            else:
                output = output[:,0,:]
            output = output.view(-1,self.n_negatives+2,output.size(1))
            queries_result = output[:,:1,:]
            docs_result = output[:,1:,:]
        return queries_result,docs_result

            

    def save(self,output_dir, tokenizer):
        self.doc_encoder.save_pretrained(output_dir)
        if not self.shared_weights:
            query_output_dir = os.path.join(output_dir,"query")
            os.makedirs(query_output_dir, exist_ok=True)
            self.query_encoder.save_pretrained(query_output_dir)
            if tokenizer:
                tokenizer.save_pretrained(query_output_dir)

class QLoRALLaMa(torch.nn.Module):

    def __init__(self, model_type_or_dir, shared_weights=True, n_negatives=-1, tokenizer=None, model_q=None, pooling='cls', lora_config_args=None, quantization_config_args=None, training=True):
        """
        output indicates which representation(s) to output ('MLM' for MLM model)
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        self.shared_weights = shared_weights  
        #self.doc_encoder = AutoModel.from_pretrained(model_type_or_dir, quantization_config=quantization_config, device_map="auto")
        self.n_negatives = n_negatives
        self.tokenizer = tokenizer
        self.training = training
        ###################################################
        # get quantization config for loading model in NF4bit
        self.quantization_config = self.get_quantization_config(quantization_config_args)
        ###################################################

        ###################################################
        # make accomodations for 4bit training
        if training:
            # prep encoder for QLoRA training
            # initialize model, update special token changes and turn it into QLoRa version
            encoder = AutoModel.from_pretrained(model_type_or_dir, quantization_config=self.quantization_config, device_map="auto")
            encoder.gradient_checkpointing_enable()
            encoder.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching, this is only for training and not inference
            # encoder.enable_input_requires_grad() # from tevatron, no documentation as to why this is needed but keeping it here, not possible for LLaMaModel..., model doesn't have this capability
            encoder = prepare_model_for_kbit_training(encoder) # preps model for QLoRA, like gradient checkpointing
            # get the lora config to load in with adapter
            self.lora_config = self.get_lora_config(lora_config_args, encoder)
            # leave get_peft_model once model is fully initialized
            # encoder = get_peft_model(encoder, self.lora_config)
        # quantized inference would save inference time, albeit at cost of efficacy, it won't quantize outputs so still full-sized
        # keeping this setting since this is how it was trained, so it makes sense to preserve the patterns learned in this setting
        else:
            # simply load the finetuned model in 4bits for 4bit inference
            # this loads LLaMa 2 into 4 bit and adds the finetuned adapters 
            # furthermore loads the tokenizer that was adapted for information retrieval, specifically PAD = UNK and CLS = EOS = </s>
            encoder, tokenizer = self.load(model_type_or_dir, self.quantization_config)
            self.tokenizer = tokenizer

        self.doc_encoder = encoder
        ###################################################

        # encoder.resize_token_embeddings(len(tokenizer)) # resize to account for sep and mask tokens
        # encoder.pad_token_id = tokenizer.pad_token_id # update configs 
        # encoder.config.mask_token_id = tokenizer.mask_token_id
        # encoder.config.sep_token_id = tokenizer.sep_token_id
        # encoder.config.cls_token_id = tokenizer.cls_token_id
        ###################################################

        self.pooling = pooling
        if shared_weights:
            self.query_encoder = self.doc_encoder
        else:
            if model_q:
                self.query_encoder = AutoModel.from_pretrained(model_q, quantization_config)
            else:
                self.query_encoder = AutoModel.from_pretrained(model_type_or_dir, quantization_config)

    def get_quantization_config(self, config_args):
        print("GETTING QUANTIZATION CONFIG")
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
                    load_in_4bit=bool(config_args["load_in_4bit"]),
                    bnb_4bit_quant_type=config_args["bnb_4bit_quant_type"],
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=bool(config_args["bnb_4bit_use_double_quant"])
        )

        return quantization_config

    def get_linear_modules(self, encoder):
        linear_modules = set()
        for i, j in encoder.named_modules():
            if "linear" in str(type(j)).lower():
                i = i.split(".")
                linear_modules.add(i[0] if len(i) == 1 else i[-1])

        return list(linear_modules)

    def get_lora_config(self, config_args, encoder):
        print("GETTING LORA CONFIG")
        linear_modules = self.get_linear_modules(encoder)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config_args["r"],
            lora_alpha=config_args["a"],
            # target_modules=["q_lin", "v_lin", "k_lin", "out_lin", "lin1", "lin2"],
            target_modules=linear_modules,
            lora_dropout=config_args["dropout"],
            bias=config_args["bias"],
            use_rslora=config_args["use_rslora"],
            use_dora=config_args["use_dora"]
        )   

        return lora_config

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.doc_encoder.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    @staticmethod
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def forward(self, **tokens):
        if not self.shared_weights:
            attention_mask = tokens["attention_mask"]
            input_ids = tokens["input_ids"]
            input_ids = input_ids.view(-1,self.n_negatives+2,input_ids.size(1))
            attention_mask = attention_mask.view(-1,self.n_negatives+2,attention_mask.size(1))
            docs_ids = input_ids[:,1:,:].reshape(-1,input_ids.size(2))
            docs_attention = attention_mask[:,1:,:].reshape(-1,attention_mask.size(2))
            queries_ids = input_ids[:,:1,:].reshape(-1,input_ids.size(2))
            queries_attention = attention_mask[:,:1,:].reshape(-1,attention_mask.size(2))

            query_result = self.query_encoder(input_ids=queries_ids,attention_mask=queries_attention)
            query_result = query_result[0]
            if self.pooling == 'mean':
                queries_result = self.mean_pooling(query_result, queries_attention)
            elif  self.pooling == 'cls': 
                queries_result = query_result[:,0,:]

            queries_result = queries_result.view(-1,1,queries_result.size(1))

            docs_result = self.doc_encoder(input_ids=docs_ids,attention_mask=docs_attention)[0]
            if self.pooling == 'mean':
                docs_result = self.mean_pooling(docs_result, queries_attention)
            else:
                docs_result = docs_result[:,0,:]
            docs_result = docs_result.view(-1,self.n_negatives+1,docs_result.size(1))

            return query_result, docs_result

        else:
            # augmenting to match tevatron implementation with SPLADE pipeline
            # https://github.com/texttron/tevatron/blob/main/examples/repllama/repllama.py

            attention_mask = tokens["attention_mask"]

            # print(f"FULL ATT MASK BEFORE RESHAPING SHAPE: {attention_mask.shape}")

            attention_mask = attention_mask.view(-1,self.n_negatives+2,attention_mask.size(1))
            docs_attention = attention_mask[:,1:,:].reshape(-1,attention_mask.size(2))
            queries_attention = attention_mask[:,:1,:].reshape(-1,attention_mask.size(2))

            # print(f"FULL ATT MASK AFTER RESHAPING SHAPE: {attention_mask.shape}")
            # print(f"DOCS ATT SHAPE: {docs_attention.shape}")
            # print(f"QUER ATT SHAPE: {queries_attention.shape}")

            input_ids = tokens["input_ids"]

            # print(f"FULL INPUT IDS BEFORE RESHAPING SHAPE: {input_ids.shape}")

            input_ids = input_ids.view(-1,self.n_negatives+2,input_ids.size(1))
            docs_ids = input_ids[:,1:,:].reshape(-1,input_ids.size(2))
            queries_ids = input_ids[:,:1,:].reshape(-1,input_ids.size(2))

            # print(f"FULL INPUT IDS AFTER RESHAPING SHAPE: {input_ids.shape}")
            # print(f"DOCS IDS SHAPE: {docs_ids.shape}")
            # print(f"QUER IDS SHAPE: {queries_ids.shape}")
        
            query_result = self.doc_encoder(input_ids=queries_ids,attention_mask=queries_attention, output_hidden_states=True)
            q_hidden = query_result.hidden_states[-1]
            q_sequence_lengths = queries_attention.sum(dim=1)
            q_last_token_indices = q_sequence_lengths - 1

            # print(f"Q_HIDDEN SHAPE: {q_hidden.shape}")
            # print(f"Q_SEQ_LENGTHS.shape: {q_sequence_lengths.shape}")
            # print(f"Q_LAST_TOKEN INDICES shape: {q_last_token_indices.shape}")

            q_reps = q_hidden[torch.arange(q_hidden.size(0)), q_last_token_indices]

            # print(f"Q REPS SHAPE BEFORE NORMALIZE: {q_reps.shape}")

            q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=-1)

            # print(f"Q REPS SHAPE AFTER NORMALIZE: {q_reps.shape}")


            docs_result = self.doc_encoder(input_ids=docs_ids, attention_mask=docs_attention, output_hidden_states=True)
            d_hidden = docs_result.hidden_states[-1]
            d_sequence_lengths = docs_attention.sum(dim=1)
            d_last_token_indices = d_sequence_lengths - 1

            # print(f"D_HIDDEN SHAPE: {d_hidden.shape}")
            # print(f"D_SEQ_LENGTHS.shape: {d_sequence_lengths.shape}")
            # print(f"D_LAST_TOKEN INDICES shape: {d_last_token_indices.shape}")

            d_reps = d_hidden[torch.arange(d_hidden.size(0)), d_last_token_indices]

            # print(f"D REPS SHAPE BEFORE NORMALIZE: {d_reps.shape}")

            d_reps = torch.nn.functional.normalize(d_reps, p=2, dim=-1)

            # print(f"D REPS SHAPE AFTER NORMALIZE: {d_reps.shape}")

            bs = attention_mask.shape[0]
            dim = d_reps.shape[-1]

            # ensure results for bs of 1 work with the loss computations
            q_reps = q_reps.view(bs, -1, dim) #(bs, 1, 4096)
            d_reps = d_reps.view(bs, -1, dim) #(bs, n_neg + 1, 4096)

            return q_reps, d_reps

    def save(self,output_dir, tokenizer):
        self.doc_encoder.save_pretrained(output_dir)
        if not self.shared_weights:
            query_output_dir = os.path.join(output_dir,"query")
            os.makedirs(query_output_dir, exist_ok=True)
            self.query_encoder.save_pretrained(query_output_dir)
            if tokenizer:
                tokenizer.save_pretrained(query_output_dir)

    # this is for loading a finetuned model using QLoRA with the trained adapters
    def load(self, model_type_or_dir, quantization_config):
        base_encoder = AutoModel.from_pretrained(model_type_or_dir, quantization_config=quantization_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir, add_eos_token=True, use_fast=True)
        # lora_config = LoraConfig.from_pretrained(model_type_or_dir)
        peft_model = PeftModel.from_pretrained(base_encoder, model_type_or_dir)

        return peft_model, tokenizer