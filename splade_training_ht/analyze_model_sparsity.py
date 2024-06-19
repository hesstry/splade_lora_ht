import os
import re

# prefix = out.out
# prefix = out.err

file_dir = "/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/training_outputs"

file_name = "params.out"

full_path = os.path.join(file_dir, file_name)

param_names = ["dthres", "dmeanthres", "qthres", "qmeanthres"]

final_thresh = {}

i = 0
match_count = 0
with open(full_path, "r") as fIn:
    while i < 100000:
        out = fIn.readline()
        for name in param_names:
            if name in out.lower():
                # print(name)
                # find name of parameter
                # total of 4 terms to match and record final values
                match_count += 1
                out = re.search(r'(.+?)\:', out)
                out = out.group(1)
                param_name = out
                # get output of the tensor
                out = fIn.readline()
                # get value for tensor
                out = re.search(r'tensor\(\[(.+?)\]', out)
                if out:
                    # print(float(out.group(1)))
                    param_value = out.group(1)
                if match_count == 4:
                    i+=1
                    match_count = 0
                if i == 99999:
                    print(f"param_name: {param_name}\nparam_val: {param_value}")
                break