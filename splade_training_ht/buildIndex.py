import json
import sys
import gzip
import numpy as np
import struct
from numpy.core.defchararray import encode
from pyfastpfor import getCodec
from tqdm import tqdm
import os

from safetensors import safe_open

def convertBinary(num):
    n = int(num)
    return struct.pack('<I', n)

def binarySequence(arr, fout):
    size = len(arr)
    fout.write(convertBinary(size))
    for i in arr:
        fout.write(convertBinary(i))

# going to heavily rework this to work with mean thresholding

if __name__ == "__main__":

    # safetensors_path = "/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/plus_mean/plus_mean_55000_100000/0_MLMTransformer/model.safetensors"

    # with safe_open(safetensors_path, framework="pt", device=0) as f:
    #     for k in f.keys():
    #         print(k)

    vocab_size = 30522

    json_path = sys.argv[1]

    posting = {}

    length = []

    # note that scaling is done before 
    scale = int(sys.argv[3])
    thres = int(scale * float(sys.argv[5])) # since thres are between 0 and 1
    mean_thresh = int(scale * float(sys.argv[6])) # also a float

    # identify which thresholding technique we are implementing, ["qd", "mean", "plus_mean"]
    thresh_type = sys.argv[7]

    print(f"CURRENT SCALE: {scale}")
    print(f"CURRENT qd THRESH: {thres}")
    print(f"CURRENT MEAN THRESH: {mean_thresh}")
    print(f"THRESHOLDING TYPE: {thresh_type}")

    if "mean" in thresh_type:
        use_mean = True
    else:
        use_mean = False

    if "plus" in thresh_type:
        plus_mean = True
    else:
        plus_mean = False

    print(f"PLUS_MEAN: {plus_mean}")
    print(f"USE MEAN: {use_mean}")

    json_path_prefix = os.path.join(json_path, "file_")

    for i in tqdm(range(int(sys.argv[4]))):
        print(i)
        for line in gzip.open("%s%d.jsonl.gz" % (json_path_prefix, i)):
            doc_dict = json.loads(line)
            id = doc_dict['id']

            vector = doc_dict['vector']

            # print(f"Vector type: {type(vector)}")
            # print(vector)

            # for k in vector:
            #     print(k)

            # print(len(vector))
            # print(len(list(vector.values())))

            # vector_mean = np.sum( np.array(list(vector.values())) ) / 30522

            # print(vector_mean)

            # exit()

            if use_mean:
                # np_vector = np.array(vector) # might be really slow /:
                vector_mean = np.sum( np.array(list(vector.values())) ) / 30522 # maybe non-zero mean thresholding in future?
                # by multiplying by mean_thresh, we've already accounted for the scaling factor
                mean_thresh_ = mean_thresh * vector_mean # = 100 * mean_thresh * vector_mean, as desired

            if plus_mean:
                curr_thresh = thres + mean_thresh_ # = 100*thresh + 100 * mean_thresh * vector_mean = 100 (thresh + mean_thresh * vector_mean)
            
            elif (not plus_mean) and (use_mean):
                curr_thresh = mean_thresh_

            elif (not plus_mean) and (not use_mean):
                curr_thresh = thres

            length_t = 0
            for k in vector:
                score = int(scale * vector[k])
                if score > int(curr_thresh):
                    length_t += 1

                    if k not in posting:
                        posting[k] = []

                    posting[k] += [id, score]
            
            length.append(length_t)

    term_id = {}
    id = 0
    for k in posting:
        term_id[k] = id
        id += 1

    with open(sys.argv[2] + '.id', 'w') as f:
        json.dump(term_id, f)

    fout_docs = open(sys.argv[2] + ".docs", 'wb')
    fout_freqs = open(sys.argv[2] + ".freqs", 'wb')
    binarySequence([len(length)], fout_docs)


    for k in tqdm(posting):
        binarySequence(posting[k][::2], fout_docs) # docIDs
        binarySequence(posting[k][1::2], fout_freqs) # score instead of freq
    fout_docs.close()
    fout_freqs.close()
    fout_sizes = open(sys.argv[2] + ".sizes", 'wb')
    binarySequence(length, fout_sizes)
    fout_sizes.close()
