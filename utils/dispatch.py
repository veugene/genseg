import argparse
from datetime import datetime
import os 
import re
import subprocess
import sys
import time


"""
A parser that collects all cluster-specific arguments. To be merged into
an experiment's parser via `parents=[dispatch_parser]`.
"""
def dispatch_argument_parser(*args, **kwargs):
    parser = argparse.ArgumentParser(*args, **kwargs)
    g_sel = parser.add_argument_group('Cluster select.')
    mutex_cluster = g_sel.add_mutually_exclusive_group()
    mutex_cluster.add_argument('--dispatch_dgx', default=False,
                               action='store_true')
    mutex_cluster.add_argument('--dispatch_ngc', default=False,
                               action='store_true')
    mutex_cluster.add_argument('--dispatch_canada', default=False,
                               action='store_true')
    g_dgx = parser.add_argument_group('DGX cluster')
    g_dgx.add_argument('--cluster_id', type=int, default=425)
    g_dgx.add_argument('--docker_id', type=str,
                       default="nvidian_general/"
                               "9.0-cudnn7-devel-ubuntu16.04_genseg:v2")
    g_dgx.add_argument('--gdx_gpu', type=int, default=1)
    g_dgx.add_argument('--gdx_cpu', type=int, default=2)
    g_dgx.add_argument('--gdx_mem', type=int, default=12)
    g_dgx.add_argument('--nfs_host', type=str, default="dcg-zfs-03.nvidia.com")
    g_dgx.add_argument('--nfs_path', type=str,
                       default="/export/ganloc.cosmos253/")
    g_ngc = parser.add_argument_group('NGC cluster')
    g_ngc.add_argument('--ace', type=str, default='nv-us-west-2')
    g_ngc.add_argument('--instance', type=str, default='ngcv1',
                       choices=['ngcv1', 'ngcv2', 'ngcv4', 'ngcv8'],
                       help="Number of GPUs.")
    g_ngc.add_argument('--image', type=str,
                       default="nvidian/lpr/"
                               "9.0-cudnn7-devel-ubuntu16.04_genseg:v2")
    g_ngc.add_argument('--source_id', type=str, default=None)
    g_ngc.add_argument('--dataset_id', type=str, default=None)
    g_ngc.add_argument('--workspace', type=str,
                       default='8CfEU-RDR_eu5BDfnMypNQ:/workspace')
    g_ngc.add_argument('--result', type=str, default="/results")
    g_cca = parser.add_argument_group('Compute Canada cluster')
    g_cca.add_argument('--account', type=str, default='rpp-bengioy',
                       choices=['rpp-bengioy', 'def-bengioy'],
                       help="Use rpp on cedar, def on graham.")
    g_cca.add_argument('--cca_gpu', type=int, default=1)
    g_cca.add_argument('--cca_cpu', type=int, default=2)
    g_cca.add_argument('--cca_mem', type=str, default='12G')
    g_cca.add_argument('--time', type=str, default='1-00:00',
                       help="Max run time (DD-HH:MM). Shorter times get "
                            "higher priority.")
    return parser


"""
Given a parser that contains _all_ of an experiment's arguments (including
the cluster-specific arguments from `dispatch_parser`), as well as a run()
method, run the experiment on the specified cluster or locally if no cluster
is specified.

NOTE: args must contain `path`.
"""
def dispatch(parser, run):
    # Get arguments.
    args = parser.parse_args()
    assert hasattr(args, 'path')
    
    # If resuming, merge with loaded arguments (newly passed arguments
    # override loaded arguments).
    if os.path.exists(os.path.join(args.path, "args.txt")):
        with open(os.path.join(args.path, "args.txt"), 'r') as f:
            saved_args = f.read().split('\n')[1:]
            args = parser.parse_args(saved_args)
    
    # Dispatch on a cluster (or run locally if none specified).
    if args.dispatch_dgx:
        _dispatch_dgx(args)
    elif args.dispatch_ngc:
        _dispatch_ngc(args)
    elif args.dispatch_canada:
        import daemon
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        daemon_log_file = open(os.path.join(args.path, "daemon.log"), 'a')
        with daemon.DaemonContext(stdout=daemon_log_file):
            _dispatch_canada_daemon(args)
    elif args.model_from is None and not os.path.exists(args.path):
        parser.print_help()
    else:
        run()


def _dispatch_dgx(args):
    pre_cmd = ("export HOME=/tmp; "
               "export ROOT=/scratch/; "
               "cd /scratch/ssl-seg-eugene; "
               "source register_submodules.sh;")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --_dispatch_dgx", "")          # Remove recursion.
    cmd = "bash -c '{} python3 {};'".format(pre_cmd, cmd)  # Combine.
    mount_point = "/scratch"
    subprocess.run(["dgx", "job", "submit",
                    "-i", str(args.docker_id),
                    "--gpu", str(args.dgx_gpu),
                    "--cpu", str(args.dgx_cpu),
                    "--mem", str(args.dgx_mem),
                    "--clusterid", str(args.cluster_id),
                    "--volume", "{}@{}:{}".format(args.nfs_path,
                                                  args.nfs_host,
                                                  mount_point),
                    "-c", cmd])


def _dispatch_ngc(args):
    pre_cmd = ("cd /repo; "
               "source register_submodules.sh;")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --_dispatch_ngc", "")          # Remove recursion.
    cmd = "bash -c '{} python3 {};'".format(pre_cmd, cmd)  # Combine.
    share_path = "/export/ganloc.cosmos253/"
    share_host = "dcg-zfs-03.nvidia.com"
    mount_point = "/scratch"
    subprocess.run(["ngc", "batch", "run",
                    "--image", args.image,
                    "--ace", args.ace,
                    "--instance", args.instance,
                    "--commandline", cmd,
                    "--datasetid", args.dataset_id,
                    "--datasetid", args.source_id,
                    "--workspace", args.workspace,
                    "--result", args.result])


def _dispatch_canada(args):
    pre_cmd = ("cd /scratch/veugene/ssl-seg-eugene\n"
               "source register_submodules.sh\n"
               "source activate genseg\n")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --dispatch_canada",   "")          # Remove recursion.
    cmd = "#!/bin/bash\n {}\n python3 {}".format(pre_cmd, cmd)  # Combine.
    out = subprocess.check_output([
                    "sbatch",
                    "--account", args.account,
                    "--gres", 'gpu:{}'.format(args.cca_gpu),
                    "--cpus-per-task", str(args.cca_cpu),
                    "--mem", args.cca_mem,
                    "--time", args.time],
                   input=cmd,
                   encoding='utf-8')
    print(out)
    return out


def _dispatch_canada_daemon(args):
    # Parse time argument to seconds.
    if   args.time.count(':')==0 and args.time.count('-')==0:
        # minutes
        time_format = "%M"
    elif args.time.count(':')==0 and args.time.count('-')==1:
        # days-hours
        time_format = "%d-%H"
    elif args.time.count(':')==1 and args.time.count('-')==0:
        # minutes:seconds
        time_format = "%M:%S"
    elif args.time.count(':')==1 and args.time.count('-')==1:
        # days-hours:minutes
        time_format = "%d-%H:%M"
    elif args.time.count(':')==2 and args.time.count('-')==0:
        # hours:minutes:seconds
        time_format = "%H:%M:%S"
    elif args.time.count(':')==2 and args.time.count('-')==1:
        # days-hours:minutes:seconds
        time_format = "%d-%H:%M:%S"
    else:
        raise ValueError("Invalid `time` format ({}).".format(args.time))
    datetime_obj = datetime.strptime(args.time, time_format)
    time_seconds = ( datetime_obj
                    -datetime(datetime_obj.year, 1, 1)).total_seconds()
    if '-' in args.time:
        # Add a day if days are specified, since days count up from 1.
        time_seconds += 24*60*60
    
    # Periodically check status of job. Relaunch on TIMEOUT.
    status = 'TIMEOUT'
    while status=='TIMEOUT':
        # Launch.
        sbatch_output = _dispatch_canada(args)
        job_id = sbatch_output.split(' ')[-1].strip(' \n\t')
        try:
            int(job_id)
        except ValueError:
            raise RuntimeError("Cannot extract job ID from `sbatch` standard "
                               "output: {}".format(sbatch_output))
        
        # Wait until the job is launched before setting a timer.
        status = _get_status_canada(job_id)
        while status=='PENDING' or status is None:
            time.sleep(10)
            status = _get_status_canada(job_id)
        
        # Wait.
        time.sleep(time_seconds)
        
        # Check status.
        status = _get_status_canada(job_id)
        
        # If the job is still RUNNING, wait until the state changes.
        # 
        # The job may continue running while the cluster waits for it
        # to exit after a TERM signal (or until it eventually KILLs the job).
        while status=='RUNNING':
            time.sleep(10)
            status = _get_status_canada(job_id)


def _get_status_canada(job_id):
    status = None
    sacct_output = subprocess.check_output(["sacct", "-j", job_id],
                                           encoding='utf-8')
    for line in sacct_output.split('\n'):
        if re.search("\s{}\s".format(job_id), line):
            # Status is in the last column : last word in the line.
            status = re.search('\s(\w+)\s*$', line).group(0).strip(' \t\n')
            break
    return status