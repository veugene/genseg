import argparse
import os
import subprocess
import re


def get_args():
    parser = argparse.ArgumentParser(description=""
        "For each experiment path, find the latest job ID and check its "
        "status.")
    parser.add_argument('experiment_paths', type=str, nargs='+')
    args= parser.parse_args()
    return args


def get_status_canada(job_id):
    status = None
    sacct_output = subprocess.check_output(["sacct", "-j", job_id],
                                           encoding='utf-8')
    for line in sacct_output.split('\n'):
        if re.search("\s{}\s".format(job_id), line):
            # Status is in the last column : last word in the line.
            status = re.search('\s(\w+)\s*$', line).group(0).strip(' \t\n')
            break
    return status


def find_latest_lock_file(path):
    if not os.path.isdir(path):
        return None
    biggest_n = None
    for fn in os.listdir(path):
        match = re.search('^lock.(\d*)$', fn)
        if match:
            n = int(match.group(1))
            if biggest_n is None:
                biggest_n = 0
            if n > biggest_n:
                biggest_n = n
    if biggest_n is None:
        return None
    return os.path.join(path, f'lock.{biggest_n}')


if __name__=='__main__':
    args = get_args()
    for path in args.experiment_paths:
        status = '?'
        lock_path = find_latest_lock_file(path)
        with open(lock_path, 'r') as lock:
            pid = int(lock.readline())
            match = re.search("(?<=Submitted batch job )[0-9].*[0-9]$",
                              lock.readline())
        if match:
            job_id = match.group(0)
            status = get_status_canada(job_id)
        print(f'{status[0]} : {path}')