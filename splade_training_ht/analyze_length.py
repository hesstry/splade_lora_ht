import matplotlib.pyplot as plt
import re

filepath = "/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/training_outputs/params.out"

dlens = []
qlens = []
i = 0
with open(filepath, "r") as fIn:
    # lines = fIn.readlines()
    # for line in lines:
    while i < 100:
        line = fIn.readline()
    # line = fIn.readline()
        if "query len" in line:
            print(line)
            out = re.search(r'tensor\((.+?)\,', line)
            out = out.group(1)
            print(out)
            print(int(out)/32)
            print()
        if "pos len" in line:
            print(line)
            out = re.search(r'tensor\((.+?)\,', line)
            out = out.group(1)
            print(out)
            print(int(out)/32)
        if "neg len" in line:
            print(line)
            out = re.search(r'tensor\((.+?)\,', line)
            out = out.group(1)
            print(out)
            print(int(out)/32)

        i += 1