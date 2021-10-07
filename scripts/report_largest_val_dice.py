import argparse
import os
import re


def get_args():
    parser = argparse.ArgumentParser(description=""
        "Get the largest validation Dice score from `val_log.txt` in every "
        "experiment directory.")
    parser.add_argument('experiment_paths', type=str, nargs='+')
    args= parser.parse_args()
    return args


def get_best_score(f):
    lines = f.readlines()
    best_score = 0
    best_l_num = 0
    l_num = 0
    regex = re.compile("(?<=val_dice\=)-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?")
    for l in lines:
        result = regex.search(l)
        if result:
            l_num += 1
            score = float(l[result.start():result.end()])
            if score < best_score:
                best_score = score
                best_l_num = l_num
    return best_score, best_l_num


if __name__=='__main__':
    args = get_args()
    for path in args.experiment_paths:
        if not os.path.isdir(path):
            continue
        try:
            f = open(os.path.join(path, "val_log.txt"), 'r')
            best_score, line_number = get_best_score(f)
            print("{} : line {} : {}".format(path, line_number, best_score))
        except:
            print("{} : None".format(path))
