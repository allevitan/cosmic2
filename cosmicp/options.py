#!/usr/bin/env python

import sys, getopt, os
from cosmicp.common import printv, color, bcolors

default_conf = os.path.join(os.path.expanduser('~')) + "/cosmicp_config/default.json" #This default.json is written there during installation
default_output_address = "127.0.0.1:50008"
default_intermediate_address = "127.0.0.1:50021" 

help =   "\nUsage: cosmicp.py [options] input.json\n\n\
\t -g   -> Perform a GPU execution, off by default.\n\
\t -c F -> Using a configuration file F. If not given, the default configuration is pulled from {}.\n\
\t -b N -> Set local batch size = N, per MPI rank. N = 20 by default.\n\
\t -m M -> Output mode. Supports M = 'disk','socket' and 'disksocket'. 'disk' (default) saves the final results into disk, \n\
\t\t\t'socket' streams the data into a xpub zmq socket, and 'disksocket' does the same but also stores the final results into disk at the end.\n\
------------------------------------------------------------------------------\n\
Streaming analysis options:\n\
\t -o ADDRESS -> Set ADDRESS as 'IP:PORT' corresponding to the address in an XSUB/XPUB router publishes all data from all MPI ranks.\n\
\t\t\tDefaults to {}\n\
\t -i ADDRESS -> Set ADDRESS as 'IP:PORT' corresponding to the intermediate address in which each MPI rank publishes their results.\n\
\t\t\tDefaults to {}\n\
\t -L -> Keep running and waiting for incoming reconstructions. Only works with an streaming reconstruction. Off by default.\n\
\n\n".format(default_conf, default_output_address, default_intermediate_address)

def parse_arguments(args, options = None):

    printv(color("\nParsing parameters...\n", bcolors.OKGREEN))

    try:
        json_file = args[0]
    except:
        raise Exception(color("\nMust provide an input JSON file\n", bcolors.FAIL))

    if options is None:
        options = {"gpu_accelerated": False,
                   "conf_file":  default_conf,
                   "batch_size_per_rank": 20,
                   "output_mode":"disk",
                   "output_address": default_output_address,
                   "intermediate_address": default_intermediate_address}

    try:
        opts, args_left = getopt.getopt(args,"hgc:b:m:o:i:L", \
                              ["gpu_accelerated", "conf_file=", "batch_size_per_rank=", "output_mode=", "output_address=", "intermediate_address=", "keep_running"])

    except getopt.GetoptError:
        printv(color(help, bcolors.WARNING))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            printv(help)
            sys.exit()
        elif opt in ("-g", "--gpu_accelerated"):
            options["gpu_accelerated"] = True
        if opt in ("-b", "--batch_size_per_rank"):
            options["batch_size_per_rank"] = int(arg)
        if opt in ("-c", "--conf_file"):
            options["conf_file"] = str(arg)
        if opt in ("-m", "--output_mode"):
            options["output_mode"] = str(arg)
        if opt in ("-o", "--output_address"):
            options["output_address"] = str(arg)
        if opt in ("-i", "--intermediate_address"):
            options["intermediate_address"] = str(arg)
        elif opt in ("-L", "--keep_running"):
            options["keep_running"] = True   


    if len(args_left) != 1:

        printv(color(help, bcolors.WARNING))
        sys.exit(2)

    else:
        options["fname"] = args_left[0]
    
    return options


    
