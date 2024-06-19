from ..colbert.modeling.inference import ModelInference
from ..colbert.modeling.colbert import ColBERT
from ..colbert.evaluation.load_model import load_model
from torch import nn
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
parser.add_argument('--dim', dest='dim', default=128, type=int)
parser.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

# Filtering-related Arguments
parser.add_argument('--mask-punctuation', dest='mask_punctuation', default=False, action='store_true')
parser.add_argument('--prune_k', dest='prune_k', default=None,type=int)
parser.add_argument('--no_context', dest='add_no_context', action='store_true')
parser.add_argument('--use_codebook', dest='use_codebook', default = None, type=str)
parser.add_argument('--hidden_dim',      type=int,   default=128,      help='code book - number of dimensions of input size')
parser.add_argument('--code_book_len',   type=int,   default=64,       help='code book M - number of codebooks')
parser.add_argument('--cluster_num',     type=int,   default=32,       help='code book K - length of a codebook')
parser.add_argument('--lsh',    action='store_true',   default=False,       help='whether use lsh in evaluation')
parser.add_argument('--position', action='store_true', default=False,       help='whether use position embedding in codebook')
parser.add_argument('--dual', action='store_true', default=False,  help='whether use uncontextual embedding in codebook')
parser.add_argument('--dual_codebook', default=None,  help='the uncontextual codebook')
parser.add_argument('--resume', dest='resume', default=False, action='store_true')
parser.add_argument('--training', default=True, action='store_true')
parser.add_argument('--resume_optimizer', dest='resume_optimizer', default=False, action='store_true')
parser.add_argument('--checkpoint', dest='checkpoint', default=None, required=False)

parser.add_argument('--lr', dest='lr', default=3e-06, type=float)
parser.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int)
parser.add_argument('--bsize', dest='bsize', default=32, type=int)
parser.add_argument('--accum', dest='accumsteps', default=2, type=int)
parser.add_argument('--amp', dest='amp', default=False, action='store_true')
parser.add_argument('--marginmse', default = 0, type = int)
parser.add_argument('--kd', default = None, type = str, help='Whether to use knowledge distillation marginMSE')
parser.add_argument('--kd_sbert', default = False, type = bool)



class DETeacher(nn.Module):
    def __init__(self, checkpoint):

        super(DETeacher, self).__init__()
        args = parser.parse_args(["--checkpoint", checkpoint])
        
        colbert, checkpoint = load_model(args)


        self.inference = ModelInference(colbert)

