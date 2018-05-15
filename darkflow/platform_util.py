import os
import subprocess
import sys
from sys import exit

'''This module implements a platform utility that exposes functions that detect platform information.'''

NUMA_NODES_STR_       = b"NUMA node(s)"
CPU_SOCKETS_STR_      = b"Socket(s)"
CORES_PER_SOCKET_STR_ = b"Core(s) per socket"
THREADS_PER_CORE_STR_ = b"Thread(s) per core"
LOGICAL_CPUS_STR_     = b"CPU(s)"

class platform:
  cpu_sockets_      = 0
  cores_per_socket_ = 0
  threads_per_core_ = 0
  logical_cpus_     = 0
  numa_nodes_       = 0

  def num_cpu_sockets(self):
    return self.cpu_sockets_

  def num_cores_per_socket(self):
    return self.cores_per_socket_

  def num_threads_per_core(self):
    return self.threads_per_core_

  def num_logical_cpus(self):
    return self.logical_cpus_

  def num_numa_nodes(self):
    return self.numa_nodes_

  def __init__(self):
    #check to see if the lscpu command is present
    lscpu_path = ''
    try:
      process = subprocess.Popen(["which", "lscpu"],
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout,stderr = process.communicate()
      if stderr:
        print ("Error: ", stderr)
        exit(1)
      else:
        lscpu_path = stdout.strip()
    except:
      print ("Error attempting to locate lscpu: ", sys.exc_info()[0])

    #get the lscpu output
    cpu_info = ''
    if lscpu_path:
      try:
        process = subprocess.Popen(lscpu_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        stdout,stderr = process.communicate()
        cpu_info = stdout.split(b"\n")
      except:
        print ("Error running lscpu: ", sys.exc_info()[0])

    #parse it
    for line in cpu_info:
#      NUMA_NODES_STR_       = "NUMA node(s)"
      if line.find(NUMA_NODES_STR_) == 0:
        self.numa_nodes_ = int(line.split(b":")[1].strip())
#      CPU_SOCKETS_STR_      = "Socket(s)"
      elif line.find(CPU_SOCKETS_STR_) == 0:
        self.cpu_sockets_ = int(line.split(b":")[1].strip())
#      CORES_PER_SOCKET_STR_ = "Core(s) per socket"
      elif line.find(CORES_PER_SOCKET_STR_) == 0:
        self.cores_per_socket_ = int(line.split(b":")[1].strip())
#      THREADS_PER_CORE_STR_ = "Thread(s) per core"
      elif line.find(THREADS_PER_CORE_STR_) == 0:
        self.threads_per_core_ = int(line.split(b":")[1].strip())
#      LOGICAL_CPUS_STR_     = "CPU(s)"
      elif line.find(LOGICAL_CPUS_STR_) == 0:
        self.logical_cpus_ = int(line.split(b":")[1].strip())
