#!/usr/bin/env python3

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_server', type=int, default=1)
parser.add_argument('--num_worker', type=int, default=3)
args = parser.parse_args()

nodename = os.environ['SLURMD_NODENAME']
node_list = os.environ['SLURM_JOB_NODELIST']
num_nodes = os.environ['SLURM_JOB_NUM_NODES']

node_prefix = node_list.split('[')[0]
_node_list = node_list.split('[')[1].split(']')[0].split(',')
node_list = []
for nodes in _node_list:
  if '-' in nodes:
    start, end = nodes.split('-')
    node_list += [i for i in range(int(start), int(end) + 1)]
  else:
    node_list += [int(nodes)]

node_list = ['%s%d' % (node_prefix, i) for i in node_list]

index = node_list.index(nodename)

server_list = [node_list[i] + ':6666' for i in range(args.num_server)]
client_list = [node_list[i] + ':6666' for i in range(args.num_server, args.num_server + args.num_worker)]

servers = ','.join(server_list)
clients = ','.join(client_list)

cmd = 'python3 ./mnist-distributed.py %s %s %s %s' % (
  servers, clients, 'ps' if index < args.num_server else 'worker', index if index < args.num_server else index - args.num_server
)

print(cmd)
os.system(cmd)
