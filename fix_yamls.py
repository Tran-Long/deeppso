import os
import glob
import yaml
import re

d = "configs/nc_nk_cases"
for f in glob.glob(f"{d}/*.yaml"):
    with open(f, "r") as infile:
        content = infile.read()
    
    # We want to parse lines like `env.training_cfg.n_cities: 20`
    # and turn them into proper dicts
    out_dict = {}
    lines = content.split('\n')
    for line in lines:
        if line.startswith('#'): continue
        if not line.strip(): continue
        k, v = line.split(':', 1)
        k = k.strip()
        v = v.strip().replace('#', '')
        # fix typo env.test.cfg
        k = k.replace('env.test.cfg', 'env.test_cfg')
        
        # parse int/float
        try:
            if v.isdigit(): v = int(v)
            else: v = float(v)
        except ValueError:
            pass # keep string
            
        parts = k.split('.')
        curr = out_dict
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                curr[part] = v
            else:
                if part not in curr:
                    curr[part] = {}
                curr = curr[part]

    with open(f, "w") as outfile:
        outfile.write("# @package _global_\n")
        yaml.dump(out_dict, outfile, default_flow_style=False)
    print(f"Fixed {f}")
