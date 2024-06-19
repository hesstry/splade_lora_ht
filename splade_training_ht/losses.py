#FROM Sentence-BERT: (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/losses/MultipleNegativesRankingLoss.py) (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/losses/MarginMSELoss.py) (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/util.py) with minimal changes.
#Original License APACHE2

from multiprocessing import reduction
from tkinter import E
from xml.etree.ElementPath import prepare_descendant
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F


def pairwise_dot_score(a: Tensor, b: Tensor):
    """
   Computes the pairwise dot-product dot_prod(a[i], b[i])
   :return: Vector with res[i] = dot_prod(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    
    return (a * b).sum(dim=-1)


def mse_scale(output, bench, target):
    """
   Computes the weighted mse loss
   """
    loss = nn.functional.mse_loss(output, target,reduce=None)
    print(loss)
    scale = -torch.sign(bench) * torch.sign(target) + 2
    scale = scale.detach()
    return torch.mean(loss * scale)
    

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

class L1:
    def __call__(self, batch_rep):
        return torch.sum(torch.abs(batch_rep), dim=-1).mean()

class L2:
    def __call__(self, batch_rep):
        return torch.sqrt(torch.sum(batch_rep * batch_rep, dim=-1)).mean()

class HOYER:
    def __call__(self, batch_rep):
        return (torch.sum(torch.abs(batch_rep), dim=-1).pow(2) / torch.sum(batch_rep.pow(2), dim=-1)).mean()

class UNIFORM: 
    def __call__(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class MarginMSELossSplade(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1, lambda_uni = 1e-2, uni_mse = False, thresholding="qd"):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSplade, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.labmda_uni = lambda_uni

        self.lambda_thres = 1

        self.FLOPS = FLOPS()
        self.L1=L1()
        self.L2=L2()
        self.HOYER = HOYER()
        self.uniform_mse = uni_mse
        self.uni = UNIFORM()
        self.relu =  nn.ReLU()

        self.thresholding = thresholding

        self.q_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.q_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.d_mean_thres = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        # self.term_thresh = nn.Parameter(torch.zeros((1, 30522)), requires_grad=True)
        self.iteration = 0

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, bm25_pos: Tensor, bm25_neg: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        """
        Three thresholding types:
            q/d denotes only using q_thresh and d_thresh

            plus_mean denotes using both q/d thresh and query/doc mean thresholding

            mean denotes only mean thresholding
        """

        thresholding = self.thresholding
        # Apply proper thresholding to queries depending on desired threshold type
        #############################################################################################################################
        if thresholding == "qd":
            embeddings_query = self.relu(embeddings_query - self.q_thres )

        elif thresholding == "plus_mean":
            q_mean = torch.mean(embeddings_query, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            embeddings_query = self.relu(embeddings_query - (self.q_thres + self.q_mean_thres * q_mean) ) # new thresh = q_thres + q_mean_thres * q_mean

        elif thresholding == "mean":
            q_mean = torch.mean(embeddings_query, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            embeddings_query = self.relu(embeddings_query - (self.q_mean_thres * q_mean) )
        #############################################################################################################################

        # Apply proper thresholding technique to documents
        #############################################################################################################################
        if thresholding == "qd":
            embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - self.d_thres ) / 0.1) - torch.erf((embeddings_pos + self.d_thres ) / 0.1 ) + 2)    
            embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - self.d_thres ) / 0.1) - torch.erf((embeddings_neg + self.d_thres ) / 0.1 ) + 2)

        elif thresholding == "plus_mean":
            dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
            embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_thres + self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
            embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_thres + self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)

        elif thresholding == "mean":
            dp_mean = torch.mean(embeddings_pos, dim=1, keepdim=True).cuda() # (bs, 1, 1)
            dn_mean = torch.mean(embeddings_neg, dim=1, keepdim=True).cuda() # (bs, num_neg, 1)
            embeddings_pos = embeddings_pos / 2.0 * (torch.erf((embeddings_pos - (self.d_mean_thres * dp_mean) ) / 0.1) - torch.erf((embeddings_pos + (self.d_mean_thres * dp_mean) ) / 0.1 ) + 2)
            embeddings_neg = embeddings_neg / 2.0 * (torch.erf((embeddings_neg - (self.d_mean_thres * dn_mean) ) / 0.1) - torch.erf((embeddings_neg + (self.d_mean_thres * dn_mean) ) / 0.1 ) + 2)
        #############################################################################################################################

        print('ITERATION NUMBER: ', self.iteration)

        print('emb query sum:', torch.sum(embeddings_query))
        print('emb pos sum:', torch.sum(embeddings_pos))
        print('emb neg sum:', torch.sum(embeddings_neg))

        # PRINTING LENGTHS WITH RELEVANT THRESHOLD
        #############################################################################################################################
        if thresholding == "qd":
            print('emb query len:', torch.sum(torch.where(embeddings_query > self.q_thres, 1, 0)))
            print('emb pos len:', torch.sum(torch.where(embeddings_pos > self.d_thres, 1, 0)))
            print('emb neg len:', torch.sum(torch.where(embeddings_neg > self.d_thres, 1, 0)))

        elif thresholding == "plus_mean":
            print('emb query len:', torch.sum(torch.where(embeddings_query > self.q_thres + q_mean * self.q_mean_thres, 1, 0)))
            print('emb pos len:', torch.sum(torch.where(embeddings_pos > self.d_thres + dp_mean * self.d_mean_thres, 1, 0)))
            print('emb neg len:', torch.sum(torch.where(embeddings_neg > self.d_thres + dn_mean * self.d_mean_thres, 1, 0)))

        elif thresholding == "mean":
            print('emb query len:', torch.sum(torch.where(embeddings_query > q_mean * self.q_mean_thres, 1, 0)))
            print('emb pos len:', torch.sum(torch.where(embeddings_pos > dp_mean * self.d_mean_thres, 1, 0)))
            print('emb neg len:', torch.sum(torch.where(embeddings_neg > dn_mean * self.d_mean_thres, 1, 0)))
        #############################################################################################################################

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        margin_pred = scores_pos - scores_neg

        flops_doc = self.lambda_d*self.FLOPS(torch.cat(reps, 0))
        flops_query = self.lambda_q * self.L1(reps[0])
        
        if self.uniform_mse:
            uniform_dist = self.labmda_uni * (self.uni(embeddings_query) - self.uni(embeddings_neg)) ** 2
            return self.loss_fct(margin_pred, labels) + flops_doc + flops_query + uniform_dist
        
        rankloss = self.loss_fct(margin_pred.cuda(), labels.cuda())

        # CALCULATING THRESH LOSS FOR MAXIMIZING PARAMS
        #############################################################################################################################
        if thresholding == "qd":
            thres_loss = self.lambda_thres * (torch.sum(torch.log(1+torch.exp(-self.q_thres*5))) + torch.sum(torch.log(1+torch.exp(-self.d_thres * 5))))

        elif thresholding == "plus_mean":
            thres_loss = self.lambda_thres * (torch.sum(torch.log(1+torch.exp(-self.q_mean_thres * 5))) \
                                            + torch.sum(torch.log(1+torch.exp(-self.d_mean_thres * 5)))\
                                            + torch.sum(torch.log(1+torch.exp(-self.q_thres * 5))) \
                                            + torch.sum(torch.log(1+torch.exp(-self.d_thres * 5))))

        elif thresholding == "mean":
            thres_loss = self.lambda_thres * (torch.sum(torch.log(1+torch.exp(-self.q_mean_thres * 5))) \
                                            + torch.sum(torch.log(1+torch.exp(-self.d_mean_thres * 5))) )
        #############################################################################################################################

        # PRINTING THE RELEVANT PARAMETERS
        #############################################################################################################################
        if thresholding == "qd":
            print('qthres:', self.q_thres)
            print('dthres:', self.d_thres)
            print('thres_loss:', thres_loss)
            print('rankloss:', rankloss, flush=True)

        elif thresholding == "plus_mean":
            print('qthres:', self.q_thres)
            print('qmeanthres:', self.q_mean_thres)
            print('dthres:', self.d_thres)
            print('dmeanthres:', self.d_mean_thres)
            print('thres_loss:', thres_loss)
            print('rankloss:', rankloss, flush=True)

        elif thresholding == "mean":
            print('qmeanthres:', self.q_mean_thres)
            print('dmeanthres:', self.d_mean_thres)
            print('thres_loss:', thres_loss)
            print('rankloss:', rankloss, flush=True)
        #############################################################################################################################

        self.iteration += 1

        # return rankloss + thres_loss + flops_doc
        return rankloss + flops_query + flops_doc + thres_loss

class MultipleNegativesRankingLossSplade(nn.Module):
    def __init__(self, model, scale: float = 1.0, similarity_fct = dot_score, lambda_d=0.0008, lambda_q=0.0006):
        super(MultipleNegativesRankingLossSplade, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]

        flops_doc = self.lambda_d*(self.FLOPS(embeddings_b))
        flops_query = self.lambda_q*(self.FLOPS(embeddings_a))

        return self.cross_entropy_loss(scores, labels) + flops_doc + flops_query

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__, "lambda_q": self.lambda_q, "lambda_d": self.lambda_d}

class KLDivLossSplade(nn.Module):
    """
    Compute the KL div loss 
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1, scaled = False, weight_option="default", focal=False, gamma = 2.0, alpha = 0.2):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(KLDivLossSplade, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        self.L1=L1()
        self.scaled = scaled
        self.focal = focal
        if self.scaled or self.focal:
            self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.alpha = alpha
        self.weight_option = weight_option
        self.gamma = gamma

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # sparse
        reps = [result['sentence_embedding'] for result in results]
        embeddings_query = reps[0]
        embeddings_docs = reps[1:]

        scores = torch.stack([self.similarity_fct(embeddings_query, embeddings_doc) for embeddings_doc in embeddings_docs], dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        flops_doc = self.lambda_d * self.FLOPS(torch.cat(embeddings_docs,0))
        flops_query = self.lambda_q * self.L1(embeddings_query)
        
        if self.scaled == True:
            nway = int(labels.shape[1]/2)
            losses = self.loss_fct(log_scores, labels[:,:nway])
            if not self.focal:
                if self.weight_option == "default":
                    weights = torch.stack([self.alpha/(1+torch.exp(labels[:,i]-labels[:,nway])) + 1 for i in range(nway, labels.shape[1])], 1)
                elif self.weight_option == "mrr_diff":
                    weights =  torch.stack([self.alpha * (1/labels[:,i]-1/labels[:,nway]) + 1 for i in range(nway, labels.shape[1])], 1)
                loss_vector = losses * weights
            else:
                wmasks = torch.zeros_like(losses)
                wmasks[:,0] = 1
                weights =  self.gamma - self.alpha * torch.stack([(1/labels[:,i]-1/labels[:,nway]) for i in range(nway, labels.shape[1])], 1)
                
                loss_vector = losses * wmasks * (1-p_scores) ** weights + losses * (1 - wmasks) * (p_scores) ** weights
        else:
            loss_vector = self.loss_fct(log_scores, labels)
            if self.focal:
                wmasks = torch.zeros_like(loss_vector)
                wmasks[:,0] = 1
                loss_vector = loss_vector * wmasks * (1-p_scores) ** self.gamma + loss_vector * (1 - wmasks) * (p_scores) ** self.gamma

        return torch.mean(loss_vector) + flops_doc + flops_query

class KLDivLossSpladeInBatch(nn.Module):
    def __init__(self, model, similarity_fct = dot_score, lambda_d=8e-2, lambda_q=1e-1, inbatch_p = 0.0):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(KLDivLossSpladeInBatch, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.loss_inbatch = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        self.L1 = L1()
        self.de_teacher = None
        self.inbatch_p = inbatch_p
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_negs = reps[2:]

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_negs = [self.similarity_fct(embeddings_query, embeddings_neg) for embeddings_neg in embeddings_negs]

        ### flops
        flops_doc = self.lambda_d * self.FLOPS(torch.cat(reps[1:], 0))
        #flops_query = self.lambda_q*(self.FLOPS(embeddings_query))
        flops_query = self.lambda_q * self.L1(embeddings_query)

        ### hard negative kldiv
        scores = torch.stack([torch.diagonal(scores_pos, 0)] + [torch.diagonal(scores_neg, 0) for scores_neg in scores_negs], dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        loss_vector = self.loss_fct(log_scores, labels)
        
        # inbatch loss
        pred_inbatch = torch.cat([scores_pos] + [scores_neg for scores_neg in scores_negs],1)
        labels_inbatch = torch.tensor(range(len(pred_inbatch)), dtype=torch.long, device=pred_inbatch.device)  
       
        loss_inbatch = self.loss_inbatch(pred_inbatch, labels_inbatch)
        
        return torch.mean(loss_vector) + loss_inbatch * self.inbatch_p + flops_doc + flops_query



class MarginMSELossSpladeInBatch(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = dot_score, lambda_d=8e-2, lambda_q=1e-1, de_teacher = None):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSpladeInBatch, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct_pair = nn.MSELoss()
        if de_teacher is not None:
            self.loss_inbatch = nn.MSELoss()
        else:
            self.loss_inbatch = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        if de_teacher is not None:
            self.de_teacher = de_teacher
        else:
            self.de_teacher = None

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, inbatch_p = 0):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]
        #print(labels.shape)
        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)


        ### if there is a dual encoder teacher
        if self.de_teacher is not None:
            embeddings_query_teacher = self.de_teacher.inference.query(sentence_features[0]['input_ids'], sentence_features[0]['attention_mask'])
            embeddings_pos_teacher = self.de_teacher.inference.doc(sentence_features[1]['input_ids'], sentence_features[1]['attention_mask'])
            embeddings_neg_teacher = self.de_teacher.inference.doc(sentence_features[2]['input_ids'], sentence_features[2]['attention_mask'])
            
            iblabels = []
            for emb_query_teacher in embeddings_query_teacher:
                emb_q = torch.transpose(emb_query_teacher.unsqueeze(0), 1,2)
                pos_scores = self.de_teacher.inference.score(emb_q, embeddings_pos_teacher)
                neg_scores = self.de_teacher.inference.score(emb_q, embeddings_neg_teacher)
                iblabels.append(torch.cat([pos_scores, neg_scores])) #1x2B
            labels_inbatch = torch.stack(iblabels)
            #print("labels_inbatch", labels_inbatch)
            
        ### flops
        flops_doc = self.lambda_d*self.FLOPS(torch.cat(reps,0))
        flops_query = self.lambda_q*(self.L1(embeddings_query))
        ### hard negative mse
        loss_pair = self.loss_fct_pair(torch.diagonal(scores_pos, 0) - torch.diagonal(scores_neg, 0), labels[:,0] - labels[:,1])
        #print("labels", labels)
        
        # inbatch loss
        pred_inbatch = torch.cat([scores_pos,scores_neg],1)
        #print("pred", pred_inbatch)
        if self.de_teacher is None:
            labels_inbatch = torch.tensor(range(len(pred_inbatch)), dtype=torch.long, device=pred_inbatch.device)  
        else:
            pred_inbatch = torch.diagonal(pred_inbatch, 0).unsqueeze(1) - pred_inbatch
            labels_inbatch = torch.diagonal(labels_inbatch, 0).unsqueeze(1) - labels_inbatch
            labels_inbatch = labels_inbatch.to(pred_inbatch.device)
        
        loss_inbatch = self.loss_inbatch(pred_inbatch, labels_inbatch)
        
        if inbatch_p is None:
            # upweight uncertain results
            pred_probs = F.softmax(pred_inbatch, dim=-1)
            pred_entropy = - torch.sum(pred_probs * torch.log(pred_probs + 1e-6), dim=1)
            instance_weight = pred_entropy / torch.log(torch.ones_like(pred_entropy) * pred_inbatch.size(1))
            instance_weight = instance_weight.detach().mean()
            results =  2 * (1 - instance_weight) * loss_pair + 2 * instance_weight * loss_inbatch + flops_doc + flops_query
            #print("loss pair", loss_pair) #tensor(505525.8438, device='cuda:0', grad_fn=<MseLossBackward>)
            #print("loss inbatch", loss_inbatch) # tensor(614.6311, device='cuda:0', grad_fn=<NllLossBackward>)
            #print(pred_entropy)
            #print(instance_weight)
            return results

        return loss_pair + loss_inbatch * inbatch_p + flops_doc + flops_query


############ Colbert Loss##############
class MarginMSELossColBERTWithDense(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, dense_weight=0.0, scaled = False, alpha = 0.2, weight_option="default"):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossColBERTWithDense, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct_pair = nn.MSELoss()
        self.dense_weight = dense_weight
        self.scaled = scaled
        self.alpha = alpha
        self.weight_option = weight_option

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        # sentence_features: query, positive passage, negative passage
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)
        token_rep_pos =  token_reps[1] * masks[1].unsqueeze(-1)
        token_rep_pos = torch.nn.functional.normalize(token_rep_pos)
        token_rep_neg = token_reps[2] * masks[2].unsqueeze(-1)
        token_rep_neg = torch.nn.functional.normalize(token_rep_neg)

        dense_scores_pos = (token_rep_query @ token_rep_pos.permute(0,2,1)).max(2).values.sum(1)
        dense_scores_neg = (token_rep_query @ token_rep_neg.permute(0,2,1)).max(2).values.sum(1)

        # cls
        clss = [result['cls'] for result in results]
        cls_query = clss[0]
        cls_pos =  clss[1]
        cls_neg = clss[2]
        cls_scores_pos = self.similarity_fct(cls_query, cls_pos)
        cls_scores_neg = self.similarity_fct(cls_query, cls_neg)
        preds = self.dense_weight * cls_scores_pos +  dense_scores_pos - self.dense_weight * cls_scores_neg - dense_scores_neg
        
        if self.scaled:
            if self.weight_option == "default":
                weight = self.alpha/(1+torch.exp(labels[:,2]-labels[:,1])) + 1
            elif self.weight_option == "mrr_diff":
                weight = self.alpha * (1/labels[:,2]-1/labels[:,1]) + 1
            loss_pair = (weight * (preds - labels[:,0]) ** 2).mean()
        else:
            loss_pair = self.loss_fct_pair(preds, labels)
       
        return loss_pair 


class KLDivLossColBERT(nn.Module):
    def __init__(self, model, scaled = False, alpha = 0.2, weight_option="default", focal=False, gamma = 2.0):
        super(KLDivLossColBERT, self).__init__()
        self.model = model
        self.scaled = scaled
        self.focal = focal
        if self.scaled or self.focal:
            self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.alpha = alpha
        self.weight_option = weight_option
        self.gamma = gamma

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            token_rep = torch.nn.functional.normalize(token_rep)
            token_scores.append((token_rep_query @ token_rep.permute(0,2,1)).max(2).values.sum(1))
        
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        if self.scaled == True:
            nway = int(labels.shape[1]/2)
            losses = self.loss_fct(log_scores, labels[:,:nway])
            if not self.focal:
                if self.weight_option == "default":
                    weights = torch.stack([self.alpha/(1+torch.exp(labels[:,i]-labels[:,nway])) + 1 for i in range(nway, labels.shape[1])], 1)
                elif self.weight_option == "mrr_diff":
                    weights =  torch.stack([self.alpha * (1/labels[:,i]-1/labels[:,nway]) + 1 for i in range(nway, labels.shape[1])], 1)
                loss_vector = losses * weights
            else:
                wmasks = torch.zeros_like(losses)
                wmasks[:,0] = 1
                weights =  self.gamma - self.alpha * torch.stack([(1/labels[:,i]-1/labels[:,nway]) for i in range(nway, labels.shape[1])], 1)
                
                loss_vector = losses * wmasks * (1-p_scores) ** weights + losses * (1 - wmasks) * (p_scores) ** weights
        else:
            loss_vector = self.loss_fct(log_scores, labels)
            if self.focal:
                wmasks = torch.zeros_like(loss_vector)
                wmasks[:,0] = 1
                loss_vector = loss_vector * wmasks * (1-p_scores) ** self.gamma + loss_vector * (1 - wmasks) * (p_scores) ** self.gamma


        return torch.mean(loss_vector)

class MarginKLDivLossColBERT(nn.Module):
    def __init__(self, model, similarity_fct = pairwise_dot_score, scaled = False, prf=False, alpha = 0.2, weight_option="default"):
        super(MarginKLDivLossColBERT, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scaled = scaled
        if self.scaled:
            self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.prf = prf
        self.alpha = alpha
        self.weight_option = weight_option

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]

        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)
        
        if self.prf:
            token_rep_query_prf = torch.nn.functional.normalize(token_reps[1], p=2, dim=2)
            idx_pos = 2
        else:
            idx_pos = 1
            
        token_rep_pos =  token_reps[idx_pos] * masks[idx_pos].unsqueeze(-1)
        token_rep_pos = torch.nn.functional.normalize(token_rep_pos)
        dense_scores_pos = (token_rep_query @ token_rep_pos.permute(0,2,1)).max(2).values.sum(1)
        
        if self.prf:
            dense_scores_pos_prf = (token_rep_query_prf @ token_rep_pos.permute(0,2,1)).max(2).values.sum(1)
            dense_scores_pos = dense_scores_pos + 0.1 * dense_scores_pos_prf
            
        token_scores = []
        
        for idx in range(idx_pos + 1, len(token_reps)):
            token_neg_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            token_neg_rep = torch.nn.functional.normalize(token_neg_rep)
            
            dense_scores_neg = (token_rep_query @ token_neg_rep.permute(0,2,1)).max(2).values.sum(1)
            if self.prf:
                dense_scores_neg_prf = (token_rep_query_prf @ token_neg_rep.permute(0,2,1)).max(2).values.sum(1)
                dense_scores_neg = dense_scores_neg + 0.1 * dense_scores_neg_prf
                
            token_scores.append(dense_scores_neg - dense_scores_pos)
        
        scores = torch.stack(token_scores, dim=1)
        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
        
       
        if self.scaled == True:
            nway = int((labels.shape[1]-1)/2)
            losses = self.loss_fct(log_scores, labels[:,:nway])
            if self.weight_option == "default":
                weights = torch.stack([self.alpha/(1+torch.exp(labels[:,i]-labels[:,nway])) + 1 for i in range(nway + 1, labels.shape[1])], 1)
            elif self.weight_option == "mrr_diff":
                weights =  torch.stack([self.alpha * (1/labels[:,i]-1/labels[:,nway]) + 1 for i in range(nway + 1, labels.shape[1])], 1)
                
            return torch.mean(losses * weights)
        
        return torch.mean(self.loss_fct(log_scores, labels))

class MultipleNegativesRankingLossColBERT(nn.Module):
    def __init__(self, model, scaled: bool = False, prf = False, alpha = 0.2,weight_option='default'):
        super(MultipleNegativesRankingLossColBERT, self).__init__()
        self.model = model
        self.scaled = scaled
        if self.scaled:
            self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss()
            
        self.prf = prf
        self.alpha = alpha
        self.weight_option = weight_option


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], wlabels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)
        
        if self.prf:
            token_rep_query_prf = torch.nn.functional.normalize(token_reps[1], p=2, dim=2)
            idx_pos = 2
        else:
            idx_pos = 1
            
        token_scores = []
        for idx in range(idx_pos, len(token_reps)):
            token_neg_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            token_neg_rep = torch.nn.functional.normalize(token_neg_rep)
            
            dense_scores_neg = (token_rep_query @ token_neg_rep.permute(0,2,1)).max(2).values.sum(1)
            if self.prf:
                dense_scores_neg_prf = (token_rep_query_prf @ token_neg_rep.permute(0,2,1)).max(2).values.sum(1)
                dense_scores_neg = dense_scores_neg + 0.1 * dense_scores_neg_prf
              
            token_scores.append(dense_scores_neg)
        
        token_scores = torch.stack(token_scores, dim=1)    
        labels = torch.zeros(token_scores.shape[0], device=token_scores.device, dtype=torch.long)  # Example a[i] should match with b[i]
        
       
        if self.scaled:
            if self.weight_option == "default":
                weight = torch.cat([self.alpha/(1+torch.exp(wlabels[:,i]-wlabels[:,0])) + 1 for i in range(1, wlabels.shape[1])])
            elif self.weight_option == "mrr_diff":
                weight =  torch.cat([self.alpha * (1/labels[:,i]-1/labels[:,1]) + 1 for i in range(1, wlabels.shape[1])])
            
            losses = self.cross_entropy_loss(token_scores, labels)
            loss_pair = (weight * losses).mean()
        else:
            loss_pair = self.cross_entropy_loss(token_scores, labels)
       
        return loss_pair

    def get_config_dict(self):
        return {'scale': self.scaled}

class KLDivLossColBERTInBatch(nn.Module):
    def __init__(self, model, inbatch_p = 0.0):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(KLDivLossColBERTInBatch, self).__init__()
        self.model = model
        self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.loss_inbatch = nn.CrossEntropyLoss()
        self.inbatch_p = inbatch_p

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]

        # token kl
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_reps[idx] = token_reps[idx] * masks[idx].unsqueeze(-1)
            token_reps[idx] = torch.nn.functional.normalize(token_reps[idx])
            token_scores.append((token_rep_query @ token_reps[idx].permute(0,2,1)).max(2).values.sum(1))
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)
        loss_vector = self.loss_fct(log_scores, labels)

        # inbatch loss
        token_q_scores_ib = []
        for qindex in range(token_rep_query.shape[0]): #batchQ
            ib_scores = []
            for idx in range(1, len(token_reps)): #nway + 1
                ib_scores.append((token_rep_query[qindex:(qindex+1)] @ token_reps[idx].permute(0,2,1)).max(2).values.sum(1))
            token_q_scores_ib.append(torch.cat(ib_scores)) #nway+1 * batchD

        pred_inbatch = torch.stack(token_q_scores_ib,0) # batchQ * nway,batchD 
        labels_inbatch = torch.tensor(list(range(len(pred_inbatch))), dtype=torch.long, device=pred_inbatch.device)  
        loss_inbatch = self.inbatch_p * self.loss_inbatch(pred_inbatch, labels_inbatch)

        return torch.mean(loss_vector) + loss_inbatch

############### CE loss ######################
class MarginKLDivLossCE(nn.Module):
    def __init__(self, model, scaled = False, alpha = 0.2):
        super(MarginKLDivLossCE, self).__init__()
        self.model = model
        self.scaled = scaled
        if self.scaled:
            self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.alpha = alpha
        
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, scaled = False):
        reps = [self.model(sentence_feature)['cls'] for sentence_feature in sentence_features]

        scores  = torch.cat([reps[0] - x for x in reps[1:]], dim=1)        
        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
        
        if self.scaled == True:
            nway = int((labels.shape[1]+1)/2)
            losses = self.loss_fct(log_scores, labels[:,:nway])
            weights = torch.stack([self.alpha/(1+torch.exp(labels[:,i]-labels[:,nway])) + 1 for i in range(nway + 1, labels.shape[1])], 1)

            return torch.mean(losses * weights)
     
        return torch.mean(self.loss_fct(log_scores, labels[:,:nway]))


class MultipleNegativesRankingLossCE(nn.Module):
    def __init__(self, model):
        super(MultipleNegativesRankingLossCE, self).__init__()
        self.model = model
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['cls'] for sentence_feature in sentence_features]
        scores  = torch.cat(reps, dim=1)
        #scores = torch.cat([reps[0] - reps[i] for i in range(1, len(reps))], dim=1)
        
        class_labels = torch.tensor([0] * scores.shape[0], dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]

        return self.cross_entropy_loss(scores, class_labels)

    def get_config_dict(self):
        return 


class MarginMSELossSpladeAdapt(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_uni = 1e-2):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSpladeAdapt, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
        self.uni = UNIFORM()
        self.lambda_uni = lambda_uni

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        embeddings_query = self.model(sentence_features[0])['sentence_embedding'] 
        embeddings_pos = self.model[0](sentence_features[1])['sentence_embedding'] 
        embeddings_neg = self.model[0](sentence_features[2])['sentence_embedding'] 

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        margin_pred = scores_pos - scores_neg

        overall_loss = self.loss_fct(margin_pred, labels)
        uni_d = self.uni(torch.nn.functional.normalize(embeddings_pos,dim=1))
        uni_q = self.uni(torch.nn.functional.normalize(embeddings_query,dim=1))

        uniform_dist = self.lambda_uni * (uni_q - uni_d) ** 2
        print(f"marginmse: {overall_loss}, unimse: {uniform_dist}")
        overall_loss +=  uniform_dist
 
        return overall_loss


