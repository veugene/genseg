import argparse
from functools import partial
import multiprocessing
import os
import subprocess
import time


dir_list = [
    "benign_without_callbacks/benign_without_callback_01",
    "benign_without_callbacks/benign_without_callback_02",
    "benigns/benign_01",
    "benigns/benign_02",
    "benigns/benign_03",
    "benigns/benign_04",
    "benigns/benign_05",
    "benigns/benign_06",
    "benigns/benign_07",
    "benigns/benign_08",
    "benigns/benign_09",
    "benigns/benign_10",
    "benigns/benign_11",
    "benigns/benign_12",
    "benigns/benign_13",
    "benigns/benign_14",
    "cancers/cancer_01",
    "cancers/cancer_02",
    "cancers/cancer_03",
    "cancers/cancer_04",
    "cancers/cancer_05",
    "cancers/cancer_06",
    "cancers/cancer_07",
    "cancers/cancer_08",
    "cancers/cancer_09",
    "cancers/cancer_10",
    "cancers/cancer_11",
    "cancers/cancer_12",
    "cancers/cancer_13",
    "cancers/cancer_14",
    "cancers/cancer_15"]

ftp_path = "ftp://figment.csee.usf.edu/pub/DDSM/cases/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_root", type=str,
                        help="root directory into which to save data")
    parser.add_argument("--num_concurrent_downloads", type=int, default=8,
                        help="number of concurrent downloads to run")
    args = parser.parse_args()
    return args


def get_dir(url, save_root):
    print("getting {}".format(url))
    success = False
    while not success:
        result = subprocess.run(["wget",
                                 "-rc",
                                 "--retry-connrefused",
                                 "--directory-prefix={}".format(save_root),
                                 "{}".format(url)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            result.check_returncode()
            success = True
        except subprocess.CalledProcessError:
            success = False
            print("restarting wget for {} in 5 seconds".format(url))
            time.sleep(5)
    print("DONE getting {}".format(url))


if __name__=="__main__":
    args = parse_args()
    url_list = [ftp_path+path for path in dir_list]
    if not os.path.exists(args.save_root):
        os.makedirs(save_root)
    with multiprocessing.Pool(args.num_concurrent_downloads) as pool:
        pool.map(partial(get_dir, save_root=args.save_root),
                 url_list)