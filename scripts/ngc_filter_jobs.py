import argparse
import subprocess


def get_args():
    parser = argparse.ArgumentParser(description="Filter NGC job list.")
    parser.add_argument('--status', type=str, default='RUNNING',
                        choices=['RUNNING', 'FAILED', 'KILLED_BY_USER',
                                 'FINISHED_SUCCESS', 'TASK_LOST'])
    args = parser.parse_args()
    return args


def scrub(batch_list, status):
    jobs = []
    for line in batch_list.split('\n'):
        segments = line.replace(' ','').split('|')
        if len(segments)!=7:
            # Not a useful line.
            continue
        job_id = segments[1]
        if len(job_id)==0:
            # No job ID on this line; skip.
            continue
        try:
            int(job_id)
        except ValueError:
            # Not a number.
            continue
        if segments[3]==status:
            jobs.append(job_id)
    return jobs



if __name__=='__main__':
    args = get_args()
    output = subprocess.check_output(["ngc batch list"], shell=True)
    jobs = scrub(output.decode('utf-8'), args.status)
    print("\n".join(jobs))
