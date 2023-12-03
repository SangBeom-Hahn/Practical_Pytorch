from pathlib import Path
import json
from collections import OrderedDict

def read_json(cfg_fname):
    fname = Path(cfg_fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook = OrderedDict)
    
def get_opt_name(flags):
    for flg in flags:
        if(flg.startswith("--")):
            return flg.replace("--", '')
    return flags[0].replace("--", "")

def update_config(config, modification):
    if(modification is None):
        return config
    
def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)