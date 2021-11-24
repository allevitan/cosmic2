#!/usr/bin/env python

import sys
import os
import h5py
import zmq
from cosmicp.options import parse_arguments
from cosmicp.common import rank, size, mpi_enabled, printd, printv, set_visible_device, complete_metadata, color, bcolors
import socket

if __name__ == '__main__':
    args = sys.argv[1:]

    options = parse_arguments(args)

    if options["gpu_accelerated"]:
        device_order, visible_devices, n_gpus = set_visible_device(rank)

        printd(color("Number of GPUs on this host = " + str(n_gpus), bcolors.HEADER))
        printd(color("GPU devices occupancy order: " + str(device_order), bcolors.HEADER))
        printd(color("GPU visible devices for this process = " + str(visible_devices), bcolors.HEADER))

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        printv(color("\r Running on CPU, enable -g option for a GPU execution", bcolors.HEADER))


    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" #This prevents JAX from taking over the whole device memory

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #this removes some log messages from tensorflow that can be pretty annoying

    from absl import logging
    logging.set_verbosity(logging.ERROR) #This does removes some annoying warnings from JAX


    import numpy as onp
    import jax.numpy as np
    import jax
    import cosmicp.diskIO as diskIO
    import cosmicp.preprocessor as preprocessor

    from cosmicp.diskIO import frames_out, map_tiffs, read_metadata_hdf5
    from cosmicp.preprocessor import prepare, process, save_results, receive_metadata, subscribe_to_socket, xsub_xpub_router, publish_to_socket, send_metadata
    from timeit import default_timer as timer

    network_metadata = {}

    #See if we have a file or an ip address
    try:
        socket.inet_aton(options["fname"].split(":")[0]) #We need the split to remove the port number
        network_metadata["input_address"] = options["fname"]
        printv(color("\nProcessing data from socket\n", bcolors.OKGREEN))
    except OSError:
        printv(color("\nProcessing data from disk\n", bcolors.OKGREEN))
        pass


    #data coming from socket or into a socket
    if "input_address" in network_metadata or options["output_mode"] != "disk":

        network_metadata["context"] = zmq.Context()

    #data coming from socket
    if "input_address" in network_metadata:

        network_metadata["input_socket"] = subscribe_to_socket(network_metadata)

        metadata = receive_metadata(network_metadata)
        metadata = complete_metadata(metadata, options["conf_file"])

        dark_frames, exp_frames = [],[]
        print(metadata)

    elif options["fname"].endswith('.json'):
        #fname = "/cosmic-dtn/groups/cosmic/Data/2021/09/210916/210916006/210916006_002_info.json"
        metadata = diskIO.read_metadata(options["fname"])
        metadata = complete_metadata(metadata, options["conf_file"])

        print(metadata)

        dark_frames = diskIO.read_dark_data(metadata, options["fname"])
        ##########   
        base_folder = os.path.split(options["fname"])[:-1][0] + "/" 
        base_folder += os.path.basename(os.path.normpath(metadata["exp_dir"]))
        ##########
        exp_frames = map_tiffs(base_folder)

    elif options["fname"].endswith('.h5'):
        #fname = "/cosmic-dtn/groups/cosmic/Data/2021/09/210916/210916003/raw_data.h5"

        metadata = read_metadata_hdf5(options["fname"])
        #metadata = diskIO.read_metadata("/cosmic-dtn/groups/cosmic/Data/2021/09/210923/210923044/210923044_002_info.json")

        metadata = complete_metadata(metadata, options["conf_file"])

        print(metadata)

        f = h5py.File(options["fname"], 'r')

        dark_frames = f["entry_1/data_1/dark_frames"]
        exp_frames = f["entry_1/data_1/exp_frames"]

    
    if options["output_mode"] != "disk":

        #This will be published so that a downstream process connects
        network_metadata["output_address"] = options["output_address"] 
        #Threads of this process will publish to this address, and a router subscribes here
        network_metadata["intermediate_address"] = options["intermediate_address"]

        network_metadata["intermediate_socket"] = publish_to_socket(network_metadata)

        #rank 0 sets up the xsub and xpub router
        if rank == 0:
            xsub_xpub_router(network_metadata)

    metadata, background_avg, received_exp_frames = prepare(metadata, dark_frames, exp_frames, network_metadata)

    if options["output_mode"] != "disk" and rank == 0:
        send_metadata(network_metadata, metadata)

    out_data, my_indexes = process(metadata, exp_frames, background_avg, options["batch_size_per_rank"], received_exp_frames, network_metadata)

    if options["fname"].endswith('.h5'):
        f.close

    #In socket mode we don't save the final results
    if options["output_mode"] != "socket":
        save_results(options["fname"], metadata, out_data, my_indexes, metadata["translations"].shape[0])

    #if network_metadata != {} and rank == 0:
    #    network_metadata["context"].destroy()


  

