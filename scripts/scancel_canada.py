import argparse
import os
import subprocess


def get_args():
    parser = argparse.ArgumentParser(description="Run scancel on all jobs "
        "associated with each experiment path provided as args.")
    parser.add_argument('experiment_paths', type=str, nargs='+')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_args()
    for path in args.experiment_paths:
        print(path)
        if not os.path.exists(path):
            raise ValueError("path not found: {}".format(path))
        daemon_path = os.path.join(path, 'daemon_log.txt')
        if not os.path.exists(daemon_path):
            print("daemon_log.txt not found in {}")
            continue
        f = open(daemon_path, 'r')
        job_id_list = []
        for line in f:
            val = line.strip('\n\t ').split(' ')[-1]
            try:
                job_id = int(val)
                job_id_list.append(job_id)
            except ValueError:
                pass
        for job_id in job_id_list:
            print("scancel {}".format(job_id))
            subprocess.run(["scancel", str(job_id)])
