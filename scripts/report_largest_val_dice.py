import argparse
import os
import re
import shutil


def get_args():
    parser = argparse.ArgumentParser(description=""
        "Get the largest validation Dice score from `val_log.txt` in every "
        "experiment directory.")
    parser.add_argument('experiment_paths', type=str, nargs='+')
    parser.add_argument('--copy_images', type=str, default=None,
                        help="Copy the image outputs corresponding to the "
                             "best epoch into this directory.")
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


def copy_images(copy_from, copy_to, epoch):
    from_img_dir = os.path.join(copy_from, 'images')
    to_img_dir = os.path.join(copy_to, os.path.basename(copy_from))
    if not os.path.exists(to_img_dir):
        os.makedirs(to_img_dir)
    for fn in os.listdir(from_img_dir):
        if fn.split('_')[0]==str(epoch-1):
            from_path = os.path.join(from_img_dir, fn)
            to_path = os.path.join(to_img_dir, fn)
            shutil.copyfile(from_path, to_path)


if __name__=='__main__':
    args = get_args()
    for path in args.experiment_paths:
        if not os.path.isdir(path):
            continue
        try:
            f = open(os.path.join(path, "val_log.txt"), 'r')
            best_score, line_number = get_best_score(f)
            print("{} : line {} : {}".format(path, line_number, best_score))
            if args.copy_images is not None:
                copy_images(copy_from=path,
                            copy_to=args.copy_images,
                            epoch=line_number)
        except:
            print("{} : None".format(path))
