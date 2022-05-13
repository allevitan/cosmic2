
import sys
import subprocess
import operator
import os
import numpy as np
import json

try: 
    from mpi4py import *
    from mpi4py import MPI
except ImportError: pass

import numpy as np

mpi_enabled = "mpi4py" in sys.modules

comm = None
if mpi_enabled:
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
else:
    size = 1
    rank = 0

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def color(string, c):
    return c + string + bcolors.ENDC

def printd(string):
    if mpi_enabled:
        a = color("\r MPI Rank " + str(rank) + ": ", bcolors.HEADER)
        print(a, end="", flush=True)
        print(string)
    else: 
        printv(string)

def printv(string):
    if rank == 0:
        print(string)

def gather(local, out_shape, n_elements, dtype):

    t = None

    if dtype == int or dtype == np.int32:
        t = MPI.INT
    if dtype == float or dtype == np.float32:
        t = MPI.FLOAT

    sendbuf = np.array(local, dtype = dtype)

    counts = comm.gather(n_elements)

    indexes = None

    if rank == 0:
        indexes = np.cumsum([0] + counts[:-1])
        recvbuf = np.empty(out_shape, dtype)

    else:
        recvbuf = None

    comm.Gatherv(sendbuf=sendbuf, recvbuf=[recvbuf, counts, indexes, t])

    return recvbuf

def convert_translations(translations):

    zeros = np.zeros(translations.shape[0])

    new_translations = np.column_stack((translations[:,1]*1e-6,translations[::-1,0]*1e-6, zeros)).astype('float32') 

    return new_translations

def complete_metadata(metadata, conf_file):

    defaults = json.loads(open(conf_file).read())

    #These do not change
    metadata["detector_pixel_size"] = defaults["geometry"]["psize"] * 1e-6 #30e-6  # 30 microns
    metadata["detector_distance"] = defaults["geometry"]["distance"] / 1000.  #0.121 #this is in meters

    #These here below are options, we can edit them
    metadata["final_res"] = defaults["geometry"]["resolution"]  #3e-9 #recon pixel size meters
    metadata["desired_padded_input_frame_width"] = None
    metadata["output_frame_width"] = defaults["geometry"]["shape"]  #256 # final frame width 
    metadata["translations"] = convert_translations(np.array(metadata["translations"]))
    if metadata["double_exposure"]:
        metadata["double_exp_time_ratio"] = metadata["dwell1"] // metadata["dwell2"] # time ratio between long and short exposure
    else:
        metadata["double_exp_time_ratio"] = 1
    

    return metadata

def allgather(send_buff, recv_buff):

    global size, rank, mpi_enabled


    if size > 1 and mpi_enabled:
        recv_buff = MPI.COMM_WORLD.allgather(send_buff)
    else:
        recv_buff = [send_buff]

    return recv_buff

#Setup local GPU device based on a gpu_priority integer. The lower the priority, the more available gpu this assigns.
def set_visible_device(gpu_priority):

    cuda_dir = ""
    try:
        cmd = "which nvcc"
        cuda_dir = str(subprocess.check_output(cmd.split(" "))).replace("bin/nvcc\\n'", "").replace("b'", "")
    except: pass

    cuda_dir_list = []

    cuda_dir_list = allgather(cuda_dir, cuda_dir_list)

    #printd(', '.join(str(e) for e in cuda_dir_list))

    for i in cuda_dir_list:
        if i != "":    
            cuda_dir = i
            break

    os.environ['CUDA_PATH'] = cuda_dir

    printd(color("CUDA directory: " + cuda_dir, bcolors.HEADER))

    cmd = "nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used,pstate,utilization.gpu --format=csv"
 
    result = str(subprocess.check_output(cmd.split(" ")))

    if sys.version_info <= (3, 0):
        result = result.replace(" ", "").replace("MiB", "").replace("%", "").split()[1:] # use this for python2
    else:
        result = result.replace(" ", "").replace("MiB", "").replace("%", "").split("\\n")[1:] # use this for python3


    result = [res.split(",") for res in result]
  
    if sys.version_info >= (3, 0):
        result = result[0:-1]

    result = [[int(r[0]), int(r[3])] for r in result]
    result = sorted(result, key=operator.itemgetter(1))

    n_devices = len(result)

    nvidia_device_order = [res[0] for res in result]

    visible_devices = str(nvidia_device_order[(gpu_priority % len(nvidia_device_order))])
      
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    return nvidia_device_order, visible_devices, n_devices
