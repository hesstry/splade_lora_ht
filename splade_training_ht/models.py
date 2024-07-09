#FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
#Original License APACHE2

from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import torch
from torch import Tensor

import logging

class Splade_Pooling(nn.Module):
    def __init__(self, word_embedding_dimension: int):
        super(Splade_Pooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.config_keys = ["word_embedding_dimension"]

    def __repr__(self):
        return "Pooling Splade({})"

    def get_pooling_mode_str(self) -> str:
        return "Splade"

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        sentence_embedding = torch.max(torch.log(1 + torch.relu(token_embeddings)) * attention_mask.unsqueeze(-1), dim=1).values
        features.update({'sentence_embedding': sentence_embedding})
        return features

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Splade_Pooling(**config)


class MLMTransformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = 256,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None):
        super(MLMTransformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        self.config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.auto_model = torch.nn.DataParallel(AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=self.config, cache_dir=cache_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        self.pooling = torch.nn.DataParallel(Splade_Pooling(self.get_word_embedding_dimension())) 

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        # self.soft_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def __repr__(self):
        return "MLMTransformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.module.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        features = self.pooling(features)

        return features

    def get_word_embedding_dimension(self) -> int:
            return self.auto_model.module.config.vocab_size
        
    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return MLMTransformer(model_name_or_path=input_path, **config)

class SpladeThresholding(MLMTransformer):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = 256,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None, thresholding: str = "qd", 
                 is_training: bool = False, input_type: str = None): # extra thresholding parameter

        # instantiate it the same as the MLMTransformer above
        super(SpladeThresholding, self).__init__(model_name_or_path, 
                                                max_seq_length,
                                                model_args,
                                                cache_dir,
                                                tokenizer_args,
                                                do_lower_case,
                                                tokenizer_name_or_path)

        # time to add the extra thresholding parameters
        self.q_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.q_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)

        self.thresholding = thresholding
        self.is_training = is_training

        self.input_type = input_type

        self.relu = torch.nn.ReLU()

    def forward(self, features):

        assert self.input_type is not None, "Need to specify if you're encoding a query or a document so proper thresholding can be applied"
        
        """Returns token_embeddings, cls_token"""

        # print("PRINTING INPUT FEATURES NEED TO HAVE RELEVANT DATATYPE = DICT")
        # print(features)

        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.module.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        features = self.pooling(features)

        # reps = [features["sentence_embedding"] for feature in features]

        # encode a single query or document(s)
        reps = features["sentence_embedding"]

        # print(f"GOING TO RUN INFERENCE: {self.is_training == False}", flush=True)
        # logging.info(f"GOING TO RUN INFERENCE: {self.is_training == False}")
        # print(f"GOING TO EMBED DOCUMENTS: {self.input_type == 'd'}")
        # logging.info(f"GOING TO EMBED DOCUMENTS: {self.input_type == 'd'}")
        # print(f"USING THRESHOLDING TYPE: {self.thresholding}")
        # logging.info(f"USING THRESHOLDING TYPE: {self.thresholding}")

        # FINETUNING INPUT->OUTPUT, this is the same regardless of training/inference
        if self.input_type == "q":
            # if queries, apply soft thresholding
            q_embs = self.soft_thresholding(reps, self.thresholding)
            return q_embs

        # approximate hard thresholding only during finetuning
        if self.input_type == "d" and self.is_training:
            # if documents and training, apply approximated hard thresholding
            d_embs = self.appr_hard_thresholding(reps, self.thresholding)
            return d_embs

        # INFERENCE INPUT->OUTPUT, only changes for documents

        # DOCUMENTS
        if self.input_type == "d" and not self.is_training:
            # TODO this might be incorrect, need to test
            d_embs = self.hard_thresholding(reps, self.thresholding)
            return d_embs

    def soft_thresholding(self, q_embs, thresholding):

        if thresholding == "qd":
            thresh = self.q_thres

        elif thresholding == "plus_mean":
            q_mean = torch.mean(q_embs, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            thresh = self.q_thres + self.q_mean_thres * q_mean

        elif thresholding == "mean":
            q_mean = torch.mean(q_embs, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            thresh = self.q_mean_thres * q_mean

        q_embs = self.relu(q_embs - thresh)

        return q_embs

    # first find the appropriate threshold
    # next apply the torch.erf approximate thresholding technique
    # this can be done individually to modularize the functions
    def appr_hard_thresholding(self, embs, thresholding):

        if thresholding == "qd":
            thresh = self.d_thres

        elif thresholding == "plus_mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True)
            thresh = self.d_thres + self.d_mean_thres * embs_mean

        elif thresholding == "mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True)
            thresh = self.d_mean_thres * embs_mean

        embs = embs / 2.0 * (torch.erf( ( embs - thresh ) / 0.1 ) - torch.erf( ( embs + thresh ) / 0.01 ) + 2)

        return embs

    # approximate hard thresholding
    # not implemented
    def apply_aht(self, embs_pos, embs_neg, thresholding):
        embs_pos = self.appr_hard_thresholding(embs_pos, self.thresholding)
        embs_neg = self.appr_hard_thresholding(embs_neg, self.thresholding)

        return embs_pos, embs_neg

    # def hehe(self, embeddings_pos, embeddings_neg, thresholding):

    #     if thresholding == "qd":
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - self.d_thres ) / 0.1) - torch.erf((embeddings_pos + self.d_thres ) / 0.1 ) + 2)    
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - self.d_thres ) / 0.1) - torch.erf((embeddings_neg + self.d_thres ) / 0.1 ) + 2)

    #     elif thresholding == "plus_mean":
    #         dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
    #         dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

    #     elif thresholding == "mean":
    #         dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
    #         dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
    #         embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
    #         embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

    #     return embeddings_pos, embeddings_neg

    def hard_thresholding(self, embs, thresholding):

        # nn.Threshold(threshold, value)
        # threshold (float) – The value to threshold at
        # value (float) – The value to replace with
        # eg:
        """
        doc_emb = model.forward(doc)
        threshold_fn = nn.Threshold(thresh, 0)
        final_output = threshold_fn(doc_emb)

        output:
            final_output[i] == 0 if doc_emb[i] <= thresh
            final_output[i] == doc_emb[i] if doc_emb[i] > thresh
        """

        print(f"GOING TO ENCODE DOCUMENTS USING HARD THRESHOLDING WITH THRESHOLDING TYPE: {thresholding}")

        if thresholding == "qd":
            threshold = nn.Threshold(self.d_thres.item(), 0)
            embs = threshold(embs)

        # threshold = d_thresh + d_mean * d_mean_thresh
        elif thresholding == "plus_mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True) # (bs, 30522, 1)
            threshold = nn.Threshold((self.d_thres + embs_mean * self.d_mean_thres).item(), 0)
            embs = threshold(embs)

        elif thresholding == "mean":
            embs_mean = torch.mean(embs, dim=1, keepdim=True) # (bs, 30522, 1)
            threshold = nn.Threshold((self.d_mean_thres * embs_mean).item(), 0)
            embs = threshold(embs)

        return embs

class MLMTransformerDense(MLMTransformer):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None, dim = 768):
        super(MLMTransformerDense, self).__init__(model_name_or_path, max_seq_length,
                model_args, cache_dir,tokenizer_args, do_lower_case,tokenizer_name_or_path)
        self.linear = nn.Linear(self.config.hidden_size, dim,bias=True)
        self.acti = nn.GELU()
        self.output = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
    
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        assert self.auto_model.module.config.output_hidden_states == True
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        
        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1

        hidden_states = output_states[all_layer_idx]
        features.update({'all_layer_embeddings': hidden_states})

        features = self.pooling(features)

        features['cls'] = self.output(self.norm(self.acti(self.linear(features['all_layer_embeddings'][-1][:,0,:]))))
        features['last_layer_embeddings'] = self.linear(features['all_layer_embeddings'][-1])

        return features

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)


def pad_mask(input_tensor, tokid, maxlen = 32):
    if input_tensor.shape[1] > maxlen:
        return input_tensor
    paddings = tokid * torch.ones([input_tensor.shape[0], maxlen - input_tensor.shape[1]]).to(input_tensor.device)
    return torch.cat([input_tensor, paddings], dim=1).to(input_tensor.dtype)

class ColBERTTransformer(MLMTransformer):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None, dim = 768):
        super(ColBERTTransformer, self).__init__(model_name_or_path, max_seq_length,
                model_args, cache_dir,tokenizer_args, do_lower_case,tokenizer_name_or_path)
        self.linear = nn.Linear(self.config.hidden_size, dim,bias=True)
        self.acti = nn.GELU()
        self.output = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
    
    def forward(self, features, padding = True):
        """Returns token_embeddings, cls_token"""
        self.auto_model.module.config.output_hidden_states = True
        
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        
        if padding: 
            trans_features['input_ids'] = pad_mask(trans_features['input_ids'], self.tokenizer.mask_token_id)
            trans_features['attention_mask'] = pad_mask(trans_features['attention_mask'], 0)

        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
            if padding:
                trans_features['token_type_ids'] =  pad_mask(trans_features['token_type_ids'], 0)
        
        output_states = self.auto_model(**trans_features, return_dict=False)
        
        features.update({'attention_mask': trans_features['attention_mask']})

        
        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
            
        hidden_states = output_states[all_layer_idx]
        features.update({'all_layer_embeddings': hidden_states})

        features['cls'] = self.output(self.norm(self.acti(self.linear(features['all_layer_embeddings'][-1][:,0,:]))))
        features['last_layer_embeddings'] = self.linear(features['all_layer_embeddings'][-1])

        return features

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

 
class CETransformer(MLMTransformer):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None, dim = 768):
        super(CETransformer, self).__init__(model_name_or_path, max_seq_length,
                model_args, cache_dir,tokenizer_args, do_lower_case,tokenizer_name_or_path)
        self.linear = nn.Linear(self.config.hidden_size, dim,bias=True)
        self.acti = nn.GELU()
        self.output = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
    
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        self.auto_model.module.config.output_hidden_states = True
        
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        
       
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
            
        output_states = self.auto_model(**trans_features, return_dict=False)
        
        features.update({'attention_mask': trans_features['attention_mask']})

        
        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
            
        hidden_states = output_states[all_layer_idx]
        features.update({'all_layer_embeddings': hidden_states})

        features['cls'] = self.output(self.norm(self.acti(self.linear(features['all_layer_embeddings'][-1][:,0,:]))))
        return features

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)


class CETransformerSeq(nn.Module):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None, dim = 768):
        super(CETransformerSeq, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        self.config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.config.num_labels = 1
        self.auto_model = torch.nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config, cache_dir=cache_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        
        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def __repr__(self):
        return "CETransformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)
    
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        self.auto_model.module.config.output_hidden_states = True
        
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        
       
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
            
        output_states = self.auto_model(**trans_features, return_dict=True)
        features.update({'attention_mask': trans_features['attention_mask']})

        
        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
        #print(output_states[all_layer_idx][-1][:,0,:].shape)
        features['cls'] = output_states.logits
        return features

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)


    def get_word_embedding_dimension(self) -> int:
            return self.auto_model.module.config.vocab_size
        
    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}


    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return CETransformerSeq(model_name_or_path=input_path, **config)

 
