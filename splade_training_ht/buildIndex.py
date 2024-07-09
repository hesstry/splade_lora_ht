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

    # vocab_size = 30522

    json_path = sys.argv[1]
    output_prefix = sys.argv[2]
    num_files = int(sys.argv[3])

    posting = {}

    length = []

    json_path_prefix = os.path.join(json_path, "file_")

    for i in tqdm(range(num_files)):
        print(i)
        for line in gzip.open("%s%d.jsonl.gz" % (json_path_prefix, i)):
            doc_dict = json.loads(line)
            id = doc_dict['id']

            vector = doc_dict['vector']

            length_t = 0

            # length can simply be found by calling len(vector) since inference handles storage of only non-zero entries AFTER proper thresholding technique was applied
            for k in vector:
                # score = int(scale * vector[k])
                # thresholding already applied during inference
                # to calculate length, simply check if score > 0
                # TODO unsure if this is even needed since it only saves non-zero terms during the inference stage
                # if score > int(curr_thresh):

                # the bottom is refactored since thresholding and scaling are applied during inference
                score = vector[k]
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

    with open(output_prefix + '.id', 'w') as f:
        json.dump(term_id, f)

    fout_docs = open(output_prefix + ".docs", 'wb')
    fout_freqs = open(output_prefix + ".freqs", 'wb')
    binarySequence([len(length)], fout_docs)


    for k in tqdm(posting):
        binarySequence(posting[k][::2], fout_docs) # docIDs
        binarySequence(posting[k][1::2], fout_freqs) # score instead of freq
    fout_docs.close()
    fout_freqs.close()
    fout_sizes = open(output_prefix + ".sizes", 'wb')
    binarySequence(length, fout_sizes)
    fout_sizes.close()
