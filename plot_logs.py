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
    args = parser.parse_args()
    return args

args = parse_args()

def parse_log_file(filename):
    dd = OrderedDict({})
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            # strip "Epoch n:"
            line = line[ line.find(":")+2 :: ]
            contents = line.split(" ")
            contents = [ elem.split('=') for elem in contents ]
            for tp in contents:
                if tp[0] not in dd:
                    dd[ tp[0] ] = []
                dd[ tp[0] ].append( float(tp[1]) )
    return dd

dd_train = parse_log_file(args.logname)
if args.key not in dd_train:
    raise Exception("Specified key ({}) was not found!".format(args.key))

plt.plot(dd_train[args.key])
plt.title(args.key)
plt.ylabel(args.key)
plt.xlabel('epoch')
plt.savefig("{}/train_{}.png".format(os.path.dirname(args.logname), args.key))
plt.close()

