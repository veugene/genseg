from datetime import datetime
import os 
import re
import subprocess
import sys


"""
A parser that collects all cluster-specific arguments. To be merged into
an experiment's parser via `parents=[dispatch_parser]`.
"""
dispatch_parser = _get_parser()


"""
Given a parser that contains _all_ of an experiment's arguments (including
the cluster-specific arguments from `dispatch_parser`), as well as a run()
method, run the experiment on the specified cluster or locally if no cluster
is specified.

NOTE: args must contain `model_from`, `path`, and `resume`.
"""
def dispatch(parser, run):
    # Get arguments.
    args = parser.parse_args()
    assert hasattr(args, 'model_from')
    assert hasattr(args, 'path')
    assert hasattr(args, 'resume')
    
    # If resuming, merge with loaded arguments (newly passed arguments
    # override loaded arguments).
    if args.resume:
        with open(os.path.join(args.resume, "args.txt"), 'r') as f:
            saved_args = f.read().split('\n')[1:]
            args = parser.parse_args(saved_args)
            setattr(args, 'resume', True)
    
    # Dispatch on a cluster (or run locally if none specified).
    if args.dispatch_dgx:
        _dispatch_dgx(args)
    elif args.dispatch_ngc:
        _dispatch_ngc(args)
    elif args.dispatch_canada:
        _dispatch_canada(args)
    elif args.model_from is None and args.resume:
        parser.print_help()
    else:
        run()


def _get_parser():
    parser = argparse.ArgumentParser()
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


def _dispatch_dgx(args):
    name = re.sub('[\W]', '_', name)         # Strip non-alphanumeric.
    pre_cmd = ("export HOME=/tmp; "
               "export ROOT=/scratch/; "
               "cd /scratch/ssl-seg-eugene; "
               "source register_submodules.sh;")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --_dispatch_dgx", "")          # Remove recursion.
    cmd = "bash -c '{} python3 {};'".format(pre_cmd, cmd)  # Combine.
    mount_point = "/scratch"
    subprocess.run(["dgx", "job", "submit",
                    "-n", name,
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
    name = re.sub('[\W]', '_', name)         # Strip non-alphanumeric.
    pre_cmd = ("cd /repo; "
               "source register_submodules.sh;")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --_dispatch_ngc", "")          # Remove recursion.
    cmd = "bash -c '{} python3 {};'".format(pre_cmd, cmd)  # Combine.
    share_path = "/export/ganloc.cosmos253/"
    share_host = "dcg-zfs-03.nvidia.com"
    mount_point = "/scratch"
    subprocess.run(["ngc", "batch", "run",
                    "--name", name,
                    "--image", args.image,
                    "--ace", args.ace,
                    "--instance", args.instance,
                    "--commandline", cmd,
                    "--datasetid", args.dataset_id,
                    "--datasetid", args.source_id,
                    "--workspace", args.workspace,
                    "--result", args.result])


def _dispatch_canada(args):
    name = re.sub('[\W]', '_', name)         # Strip non-alphanumeric.
    pre_cmd = ("cd /scratch/veugene/ssl-seg-eugene\n"
               "source register_submodules.sh\n"
               "source activate genseg\n")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --_dispatch_canada",   "")          # Remove recursion.
    cmd = "#!/bin/bash\n {}\n python3 {}'".format(pre_cmd, cmd)  # Combine.
    subprocess.run(["sbatch",
                    "--account", args.account,
                    "--job_name", name,
                    "--gres", 'gpu:{}'.format(args.cca_gpu),
                    "--cpus-per-task", str(args.cca_cpu),
                    "--mem", args.cca_mem,
                    "--time", args.time],
                   input=cmd.encode('utf-8'))