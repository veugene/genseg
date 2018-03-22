import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Parse log files")
    parser.add_argument('--logname', type=str, default="")
    parser.add_argument('--key', type=str, default="")
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()
    return args

def parse_log_file(filename):
    dd = OrderedDict({})
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            marker_idx = line.find(":")
            if marker_idx!=-1:
                line = line[marker_idx+2::]     # Strip "Epoch <num>: "
            contents = line.split(" ")
            contents = [ elem.split('=') for elem in contents ]
            for tp in contents:
                if tp[0] not in dd:
                    dd[ tp[0] ] = []
                dd[ tp[0] ].append( float(tp[1]) )
    return dd

args = parse_args()

dd = parse_log_file(args.logname)
if args.key not in dd:
    raise Exception("Specified key ({}) was not found!".format(args.key))

plt.plot(dd[args.key])
plt.title(args.key)
plt.ylabel(args.key)
plt.xlabel('epoch')
if args.outdir is None:
    out_file = "{}/{}_{}.png".format(
        os.path.dirname(args.logname),
        os.path.basename(args.logname),
        args.key)
else:
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    out_file = "{}/{}_{}.png".format(
        args.outdir,
        os.path.basename(args.logname),
        args.key)

plt.savefig(out_file)
plt.close()
