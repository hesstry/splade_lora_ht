import json
import sys
import math

print(sys.argv)

term_id_file = sys.argv[1]
f_in_path = sys.argv[2]
f_out_path = sys.argv[3]

term_id = {}

# this is the index.id file generated from the buildIndex.py script
with open(term_id_file + '.id') as f:
    term_id = json.load(f)

# f_in = open('/share/paper-submission/unicoil-index/filtered-msmarco-dev-queries.tsv')
# f_in = open('/home/yingrui/MSMARCO/queries/queries_top1000.dev.tsv.spladev2.weighted')
# f_in = open('/home/yifan/Downloads/pisa/data/queries.unicoil.sorted.tsv')
f_in = open(f_in_path)
# f_in = open('deepimpact_short.dev.tsv')
# f_in = open('/nvme/yifanq/msmarco-doc-ggt/msmarco-docdev-queries-tok.tsv')
# f_in = open('/home/yingrui/MSMARCO/queries/msmarco-test2019-queries-qrel.tsv.spladev2.weighted')
# f_in = open('/home/yifan/Downloads/dl1920/msmarco-test2019-queries-qrel-piece.tsv')
# f_in = open('/nvme/yifanq/msmarco-doc-ggt/filtered-msmarco-docdev-queries.tsv')
# f_in = open('/nvme/yifanq/bm25-t5-index/queries.bert.dev.tsv')
# f_in = open('/nvme/yifanq/bm25-t5-index/queries.bert.dev.tsv.all')

# times = {}

# OUTPUT_PREFIX = argv[1]
# ENCODED Q FILE = argv[2]
# OUTPUT_Q_ID = argv[3]
# Q_THRESH = argv[4]

thres = 0

f_out = open(f_out_path, 'w')
for line in f_in:
    line = line.strip().split('\t')
    wlist = line[1].split(" ")
    qid = line[0]
    wlist_n = []

    # this block becomes useless if thresholding is performed during inference, no need to check for non-zeros as only non-zero entries are stored
    # times = {}
    # for w in wlist:
    #     if w in term_id:
    #         if w not in times:
    #             times[w] = 0
    #         times[w] += 1

    #tt = list(times.values())
    #tt.sort()
    #thres = tt[math.floor(len(tt) * (1 - float(sys.argv[4])))]
    # thres = int(sys.argv[4])
    # thres = int(float(sys.argv[4]))
    
    for w in wlist:
        if w in term_id:
            # no need to check thresholding since this is accounted for during model inference and inference_q_SPLADE.py file
            # if times[w] > thres:
            wlist_n.append(str(term_id[w]))

        # else:
        #     print(wlist[i], "not found in the given docs, assign a new term_id")
        #     term_id[wlist[i]] = id
        #     id += 1
        #     wlist[i] = str(id)
            
    to_write = qid + ': ' + ' '.join(wlist_n)
    f_out.write(to_write)
    f_out.write("\n")
f_in.close()
f_out.close()

