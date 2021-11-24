import sys
import os
import jax.numpy as np
import jax
import jax.ops
import jax.dlpack
import scipy
import scipy.constants
import scipy.interpolate
import scipy.signal
import numpy as npo
import math
import zmq
import json
import threading
from .nexus_io import write, nexus_metadata, nexus_data, cosmic_metadata
from .fccd import imgXraw as cleanXraw
from .common import printd, printv, rank, gather, color, bcolors, comm
from .common import  size as mpi_size
from .diskIO import IO, frames_out

from timeit import default_timer as timer
from functools import partial

import msgpack
import msgpack_numpy

@jax.jit
def combine_double_exposure(data0, data1, double_exp_time_ratio, thres=3e3):

    msk=data0<thres    

    return (double_exp_time_ratio+1)*(data0*msk+data1)/(double_exp_time_ratio*msk+1)
@jax.jit
def resolution2frame_width(final_res, detector_distance, energy, detector_pixel_size, frame_width):

    hc=scipy.constants.Planck*scipy.constants.c/scipy.constants.elementary_charge
   
    wavelength = hc/energy
    padded_frame_width = (detector_distance*wavelength) /(detector_pixel_size*final_res)

    return padded_frame_width # cropped (TODO:or padded?) width of the raw clean frames

#Computes a weighted average of the coordinates, where if the image is stronger you have more weight.
@jax.jit
def center_of_mass(img, coord_array_1d):
    return np.array([np.sum(img*coord_array_1d)/np.sum(img), np.sum(img*coord_array_1d.T)/np.sum(img)])

@jax.jit
def filter_frame(frame, bbox):
    return jax.scipy.signal.convolve2d(frame, bbox, mode='same', boundary='fill')


#Interpolation around the center of mass, thus centering. This downsamples into the output frame width
@partial(jax.jit, static_argnums=2)
def shift_rescale(img, center_of_mass, out_frame_shape, scale):

    img_out = jax.image.scale_and_translate(img, [out_frame_shape, out_frame_shape], [0,1], jax.numpy.array([scale, scale]), jax.numpy.array([center_of_mass[1], center_of_mass[0]]) , method = "bilinear", antialias = False).T

    img_out*=(img_out>0)

    img_out = np.float32(img_out)
            
    img_out = np.reshape(img_out, (1, out_frame_shape, out_frame_shape))

    return img_out

@jax.jit
def split_background(background_double_exp):

    # split the average from 2 exposures:
    bkg_avg0=np.average(background_double_exp[0::2],axis=0)
    bkg_avg1=np.average(background_double_exp[1::2],axis=0)

    return np.array([bkg_avg0, bkg_avg1])


def compute_background_metadata(metadata, frames, dark_frames):

    ## get one frame to compute center

    clean_frame = [] 
    if metadata["double_exposure"]:

        background_avg = split_background(dark_frames)
        for i in range(0, frames.shape[0], 2):

            frame_exp1 = frames[i] - background_avg[0]
            frame_exp2 = frames[i + 1] - background_avg[1]
            # get clean frames
            clean_frame.append(combine_double_exposure(cleanXraw(frame_exp1), cleanXraw(frame_exp2), metadata["double_exp_time_ratio"]))
            
    else:
        background_avg = np.average(dark_frames,axis=0)
        for i in range(0, frames.shape[0], 1):

            # get clean frames
            clean_frame.append(cleanXraw(frames[i] - background_avg))

    clean_frame = npo.array(clean_frame)

    metadata["frame_width"] = clean_frame.shape[0]

    #Coordinates from 0 to frame width, 1 dimension
    xx=np.reshape(np.arange(metadata["frame_width"]),(metadata["frame_width"],1))
    yy=np.reshape(np.arange(metadata["output_frame_width"]),(metadata["output_frame_width"],1))

    # cropped width of the raw clean frames
    if metadata["desired_padded_input_frame_width"]:
        metadata["padded_frame_width"] = metadata["desired_padded_input_frame_width"]

    else:
        metadata["padded_frame_width"] = float(resolution2frame_width(metadata["final_res"], metadata["detector_distance"], metadata["energy"], metadata["detector_pixel_size"], metadata["frame_width"]))
    
    # modify pixel size; the pixel size is rescaled
    metadata["x_pixel_size"] = metadata["detector_pixel_size"] * metadata["padded_frame_width"] / metadata["output_frame_width"]
    metadata["y_pixel_size"] = metadata["x_pixel_size"]

    # frame corner
    corner_x = metadata['x_pixel_size']*metadata['output_frame_width']/2  
    corner_z = metadata['detector_distance']                
    metadata['corner_position'] = [corner_x, corner_x, corner_z]
    metadata["energy"] = metadata["energy"]*scipy.constants.elementary_charge

    #Convolution kernel
    kernel_width = np.max(np.array([np.int32(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1]))
    bbox = np.ones((kernel_width,kernel_width))

    filtered_frames = []
    for i in range(0, clean_frame.shape[0]):
        clean_frame[i] = filter_frame(clean_frame[i], bbox)

        filtered_frames.append(shift_rescale(clean_frame[i], (0,0), metadata["output_frame_width"], metadata["output_frame_width"]/metadata["padded_frame_width"])[0])

    filtered_frames = npo.array(filtered_frames)

    com = center_of_mass(filtered_frames*(filtered_frames>0), yy)

    com = npo.array(np.round(com))

    metadata["center_of_mass"] = metadata["output_frame_width"]//2 - com
    metadata["output_padded_ratio"] = metadata["output_frame_width"]/metadata["padded_frame_width"]

    return metadata, background_avg


def subscribe_to_socket(network_metadata):

    addr = 'tcp://%s' % network_metadata["input_address"]

    socket = network_metadata["context"].socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.set_hwm(2000)
    socket.connect(addr)

    return socket

def publish_to_socket(network_metadata):

    addr = 'tcp://%s' % network_metadata["intermediate_address"]

    socket = network_metadata["context"].socket(zmq.PUB)
    socket.set_hwm(2000)
    socket.connect(addr)

    return socket

def xsub_xpub_router(network_metadata):

    print("Setting up XSUB XPUB router, this will cast a thread running with the proxy router...")

    frontend_socket = network_metadata["context"].socket(zmq.XPUB)
    frontend_socket.bind('tcp://%s' % network_metadata["output_address"])

    backend_socket = network_metadata["context"].socket(zmq.XSUB)
    backend_socket.bind('tcp://%s' % network_metadata["intermediate_address"])

    th = threading.Thread(target=zmq.proxy, args = (frontend_socket, backend_socket))
    th.start()

    #zmq.proxy(frontend_socket, backend_socket)

    return th


def receive_metadata(network_metadata):

    print("Waiting for metadata...")
    metadata = json.loads(network_metadata["input_socket"].recv_string())  # blocking
    print("Received metadata ")
    print(metadata)

    return metadata

def send_metadata(network_metadata, metadata):
    print("Sending metadata to socket...")

    #We remove ndarrays here so that we can serialize all the metadata.
    metadata_plain = metadata.copy()

    metadata_plain["translations"] = metadata_plain["translations"].tolist()
    metadata_plain["center_of_mass"] = metadata_plain["center_of_mass"].tolist()

    print(metadata_plain)

    network_metadata["intermediate_socket"].send_string(json.dumps(metadata_plain))


def receive_n_frames(n_frames, network_metadata):

    frames = []

    n_received = 0

    while n_received < n_frames: 

        msg = network_metadata["input_socket"].recv()

        (number, frame) = msgpack.unpackb(msg, object_hook= msgpack_numpy.decode, use_list=False,  max_bin_len=50000000, raw=False)

        print("Received frame " + str(number))

        frames.append(frame)
        n_received += 1           

    return frames

from PIL import Image

index = 0


def prepare_from_mem(metadata, dark_frames, raw_frames):

    dark_frames = np.array(dark_frames)

    n_frames = raw_frames.shape[0]
    n_total_frames = metadata["translations"].shape[0]
    if metadata["double_exposure"]: n_total_frames *= 2

    #This takes the center of the stack as a center frame(s)
    center = int(n_total_frames)//2

    #Check this in case we are in double exposure
    if center % 2 == 1:
        center -= 1

    extra_frames = 2  #we take a couple more frames to compute the center of mass, careful not to start on an odd exposure frame here

    center_frames = []
    if metadata["double_exposure"]:
        for i in range(center, center + extra_frames + 1, 2):
            center_frames.append(raw_frames[i])
            center_frames.append(raw_frames[i + 1])
    else:
        for i in range(center, center + extra_frames + 1, 1):
            center_frames.append(raw_frames[i])

    center_frames = np.array(center_frames)

    return metadata, center_frames, dark_frames


def prepare_from_socket(metadata, network_metadata):

    print("Receiving dark frames...")
    dark_frames = receive_n_frames(metadata["dark_num_total"] * (metadata['double_exposure']+1), network_metadata)

    n_some_exp_frames = 4  

    print("Receiving some exposure frames...")
    #We need some exp frames to compute the center of mass so we take those now and keep them for later 
    some_exp_frames = receive_n_frames(n_some_exp_frames, network_metadata)

    return metadata, np.array(some_exp_frames), dark_frames

def prepare(metadata, dark_frames, raw_frames, network_metadata):

    received_exp_frames = []#this is used when reading from socket, we read some exp frames here first to compute some things so we have to keep them

    #data coming in from socket
    if "input_socket" in network_metadata:
        metadata, center_frames, dark_frames = prepare_from_socket(metadata, network_metadata)
        received_exp_frames = center_frames
        print(received_exp_frames.shape)

    #data coming from mem or from disk
    else:
        metadata, center_frames, dark_frames = prepare_from_mem(metadata, dark_frames, raw_frames)
    
    metadata, background_avg =  compute_background_metadata(metadata, center_frames, dark_frames)

    return metadata, background_avg, received_exp_frames


def prepare_filter_functions(metadata, background_avg):

    #Convolution kernel
    kernel_width = np.max(np.array([np.int32(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1]))
    kernel_box = np.ones((kernel_width,kernel_width))

    cleanXraw_vmap = jax.vmap(lambda x: cleanXraw(x - background_avg))
    cleanXraw_vmap_d1 = jax.vmap(lambda x: cleanXraw(x - background_avg[0]))
    cleanXraw_vmap_d2 = jax.vmap(lambda x: cleanXraw(x - background_avg[1]))

    combine_double_exposure_vmapf = jax.vmap(lambda x, y: combine_double_exposure(x, y, metadata["double_exp_time_ratio"]))

    #single and double exposure functions
    f_cleanframes = jax.jit(lambda x: cleanXraw_vmap(x))
    f_cleanframes_d = jax.jit(lambda x, y: combine_double_exposure_vmapf(cleanXraw_vmap_d1(x), cleanXraw_vmap_d2(y)))

    def f(clean_frame):
        filtered_frame = filter_frame(clean_frame, kernel_box)
        centered_rescaled_frame = shift_rescale(filtered_frame, metadata["center_of_mass"], metadata["output_frame_width"], metadata["output_padded_ratio"])
        return centered_rescaled_frame

    process_batch_vmapf = jax.vmap(f)

    f_all = jax.jit(lambda x: process_batch_vmapf(f_cleanframes(x)))
    f_all_d = jax.jit(lambda x, y: process_batch_vmapf(f_cleanframes_d(x, y)))

    return f_all, f_all_d


def process(metadata, raw_frames, background_avg, local_batch_size, received_exp_frames, network_metadata):

    if metadata["double_exposure"]:
        printv(color("\nProcessing the stack of raw frames as a double exposure scan...\n", bcolors.OKGREEN))
    else:
        printv(color("\nProcessing the stack of raw frames as a single exposure scan...\n", bcolors.OKGREEN))

    filter_all, filter_all_dexp = prepare_filter_functions(metadata, background_avg)

    if "input_socket" in network_metadata:
        printv(color("\r Processing a stack of frames of size: {}".format((metadata["exp_num_total"], 
                       received_exp_frames[0].shape[0], received_exp_frames[0].shape[1])), bcolors.HEADER))
        results = process_from_socket(metadata, filter_all, filter_all_dexp, received_exp_frames, network_metadata)
    else:
        printv(color("\r Processing a stack of frames of size: {}".format((raw_frames.shape[0], raw_frames[0].shape[0], raw_frames[0].shape[1])), bcolors.HEADER))
        results = process_from_disk(metadata, raw_frames, local_batch_size, filter_all, filter_all_dexp, network_metadata)

    return results


def send_socket_data(frames, indexes, min_i, max_i, network_metadata):

    print("Sending output frames buffer...")
    for i in range(min_i, max_i):
        print(i)
        print(indexes)
        print("Sending frame " + str(indexes[i]) + " to socket")

        msg = msgpack.packb((b'%d' % indexes[i], npo.array(frames[i])), default=msgpack_numpy.encode, use_bin_type=True)
        network_metadata["intermediate_socket"].send(msg)


def process_from_socket(metadata, filter_all, filter_all_dexp, received_exp_frames, network_metadata):

    input_buffer_size = 12 #How many frames are stored in each rank before actually computing them

    #How many frames are stored in each rank before sending them out to a socket, 
    #this always has to be >= than input_buffer_size// (metadata['double_exposure']+1)
    output_buffer_size = 12 

    output_index = 0
    batch = 1
    number = 0

    total_input_frames = metadata["exp_num_total"] * (metadata['double_exposure']+1)
    total_output_frames = metadata["exp_num_total"]  

    output_socket = "intermediate_socket" in network_metadata

    n_batches = total_input_frames // mpi_size

    #Here we correct if the total number of frames is not a multiple of mpi_size  
    extra_frames = total_input_frames - (n_batches * mpi_size)

    extra = 0
    if rank < extra_frames: 
        extra = 1

    out_data_shape = (n_batches * mpi_size //(metadata['double_exposure']+1) + extra, metadata["output_frame_width"], metadata["output_frame_width"])

    print(out_data_shape)

    out_data = np.empty(out_data_shape,dtype=np.float32)

    frames_buffer = [] 
    index_buffer = []

    processed_batches = 0

    #We initialize here the buffer with the exp frames we have received already
    for i in range(0,len(received_exp_frames)):
        if ((i // (metadata["double_exposure"] + 1)) % mpi_size) == rank:
            frames_buffer.append(received_exp_frames[i].astype(npo.uint16)) #We will deserialize these frames again below, so we cast back to uint16
            index_buffer.append(i // (metadata["double_exposure"] + 1))

    my_indexes = []

    #print(frames_buffer)
    print(index_buffer)

    frames_ready = len(received_exp_frames) // (metadata["double_exposure"] + 1)
    frames_sent = 0

    print("Receiving all exposure frames...")

    while number != total_input_frames - 1: 

        msg = network_metadata["input_socket"].recv()  # blocking

        (number, frame) = msgpack.unpackb(msg, object_hook= msgpack_numpy.decode, use_list=False,  max_bin_len=50000000, raw=False)  

        number = int(number)
        final_number = number // (metadata["double_exposure"] + 1)

        #Each rank takes only some frames
        if (final_number % mpi_size) == rank: 

            print("Received frame " + str(number))

            frames_buffer.append(frame)
            index_buffer.append(final_number)

 
        #after filling the buffer we do the processing, or if it is the last frame we consume the buffer too
        if len(frames_buffer) == input_buffer_size or number == total_input_frames - 1:

            print("Processing input frames buffer...")

            frames_buffer = np.array(frames_buffer)

            #print(frames_buffer)

            #if double exposure we take 1 every 2 (because indexes are duplicated as they are divided by 2 above)
            my_indexes.extend(index_buffer[::metadata["double_exposure"] + 1]) 

            n_frames_out = frames_buffer.shape[0] // (metadata['double_exposure']+1)

            if metadata["double_exposure"]:
                centered_rescaled_frames_jax = filter_all_dexp(frames_buffer[:-1:2], frames_buffer[1::2])
            else:
                centered_rescaled_frames_jax = filter_all(frames_buffer)


            print(n_frames_out)
            print(output_index)
            print(centered_rescaled_frames_jax.shape)
            print(out_data.shape)

            # TODO: 'centered_rescaled_frames_jax' picks up an additional dimension somehow, should fix this...
            out_data = jax.ops.index_update(out_data, jax.ops.index[output_index:output_index + n_frames_out, :, :], centered_rescaled_frames_jax[:,0,:,:])

            print(output_index)

            #print(out_data[output_index:output_index + n_frames_out, :, :])

            frames_buffer = []
            index_buffer = []

            output_index += n_frames_out
            processed_batches += 1

            frames_ready += n_frames_out

            print("Frames ready {}".format(frames_ready))
            print(my_indexes)

            if rank == 0:
                sys.stdout.write(color("\r Computing batch = %s of %s frames\n" %(processed_batches, n_frames_out), bcolors.HEADER))
                sys.stdout.flush()

        #Seding frames to socket
        if output_socket and (frames_ready // output_buffer_size >= 1 or number == total_input_frames - 1):

            max_index = min(frames_sent + output_buffer_size, total_output_frames)
            min_index = frames_sent

            send_socket_data(out_data, my_indexes, min_index, max_index, network_metadata)

            frames_sent += output_buffer_size
            frames_ready -= output_buffer_size
            print("{} frames sent".format(min(frames_sent, total_output_frames)))
   

    return out_data, my_indexes


def process_from_disk(metadata, raw_frames, local_batch_size, filter_all, filter_all_dexp, network_metadata):

    n_total_frames = raw_frames.shape[0]

    print("TOTAL")
    print(n_total_frames)

    #if the batch size is not even in double exposure we fix that
    if local_batch_size % 2 != 0 and metadata['double_exposure']:
        local_batch_size += 1

    #If the batch size is not given or it is too big, we set it up to give work to every rank
    if local_batch_size == None or local_batch_size * mpi_size > n_total_frames:
        local_batch_size = n_total_frames // mpi_size

    batch_size = mpi_size * local_batch_size

    printv(color("\r Using a local batch size per MPI rank = " + str(local_batch_size), bcolors.HEADER))

    #This stores the frames indexes that are being process by this mpi rank
    my_indexes = []
    n_batches = raw_frames.shape[0] // batch_size

    #Here we correct if the total number of frames is not a multiple of batch_size  
    extra = raw_frames.shape[0] - (n_batches * batch_size)

    extra_last_batch = None
    if rank * local_batch_size < extra: 
        n_batches = n_batches + 1
        n_ranks_extra = int(np.ceil(extra/local_batch_size))
        #We always overshot the batch sizes if they don't match perfectly (that is when extra % local_batch_size != 0)
        #To account for this, we need to have an index substraction (extra_last_batch) for last rank accross the ones having extra work
        if rank == n_ranks_extra - 1 and extra % local_batch_size != 0: 
            extra_last_batch = - (local_batch_size - (extra % local_batch_size)) // (metadata['double_exposure']+1)     

    n_out_frames = n_batches * local_batch_size //(metadata['double_exposure']+1)    

    out_data_shape = (n_out_frames , metadata["output_frame_width"], metadata["output_frame_width"])
    out_data = np.empty(out_data_shape,dtype=np.float32)
    frames_batch = npo.empty((local_batch_size, raw_frames[0].shape[0], raw_frames[0].shape[1]))

    #Streaming variables
    frames_ready = 0
    frames_sent = 0
    streaming_output_buffer_size = 12
    output_socket = "intermediate_socket" in network_metadata

    for i in range(0, n_batches):

        local_i = ((i * batch_size) + (rank * local_batch_size)) 

        #we handle uneven shapes here
        upper_bound = min(local_i  + local_batch_size, n_total_frames)

        local_range = range(local_i // (metadata['double_exposure']+1) , upper_bound // (metadata['double_exposure']+1))

        my_indexes.extend(local_range)

        i_s = i * local_batch_size // (metadata['double_exposure']+1)
        i_e = i_s + local_batch_size // (metadata['double_exposure']+1)

        for j in range(local_i, upper_bound) : 
            frames_batch[j % local_batch_size] = raw_frames[j][:, :]

        if metadata["double_exposure"]:
            centered_rescaled_frames_jax = filter_all_dexp(frames_batch[:-1:2], frames_batch[1::2])
        else:
            centered_rescaled_frames_jax = filter_all(frames_batch)

        # TODO: 'centered_rescaled_frames_jax' picks up an additional dimension somehow, should fix this...
        out_data = jax.ops.index_update(out_data, jax.ops.index[i_s:i_e, :, :], centered_rescaled_frames_jax[:,0,:,:])

        if rank == 0:
            sys.stdout.write(color("\r Computing batch = %s/%s " %(i+1,n_batches), bcolors.HEADER))
            sys.stdout.flush()

        frames_ready += (i_e - i_s)

        #Sending frames to socket
        if output_socket:

            if extra_last_batch is not None and i == n_batches - 1: 
                i_e += extra_last_batch #extra_last_batch is a negative offset, we add it here

            send_socket_data(out_data, my_indexes, i_s, i_e, network_metadata)

            print("{} frames sent".format(i_e))


    if rank == 0: print("\n")
    return out_data[:extra_last_batch], my_indexes


def save_results(fname, metadata, local_data, my_indexes, n_frames):

    nexus_file = cxi_file = True

    n_elements = npo.prod([i for i in local_data.shape])

    frames_gather = gather(local_data, (n_frames, local_data[0].shape[0], local_data[0].shape[1]), n_elements, npo.float32)  

    print(my_indexes)

    #we need the indexes too to map properly each gathered frame
    index_gather = gather(my_indexes, n_frames, len(my_indexes), npo.int32)

    print(index_gather)

    if rank == 0:

        #Here we generate a proper index pull map, with respect to the input
        index_gather = npo.array([ npo.where(index_gather==i)[0][0] for i in range(0,len(index_gather))])

        frames_gather[:,:,:] = frames_gather[index_gather,:,:]

        printv(color("\r Final output data size: {}".format(frames_gather.shape), bcolors.HEADER))

        print(index_gather)

        #for i in range(0, frames_gather.shape[0]):
        #    print(frames_gather[i][0:10])

        dataAve = frames_gather[()].mean(0)


        pMask = np.fft.fftshift((dataAve > 0.1 * dataAve.max()))
        probe = np.sqrt(np.fft.fftshift(dataAve)) * pMask
        probe = np.fft.ifftshift(np.fft.ifftn(probe))
        
        output_filename = os.path.splitext(fname)[:-1][0][:-5]

        if cxi_file:

            io = IO()
            cxi_filename = output_filename + "_cosmic2.cxi"

            printv(color("\nSaving cxi file: " + cxi_filename + "\n", bcolors.OKGREEN))

            #This deletes and rewrites a previous file with the same name
            try:
                os.remove(cxi_filename)
            except OSError:
                pass

            io.write(cxi_filename, metadata, data_format = io.metadataFormat) #We generate a new cxi with the new data

            data_shape = frames_gather.shape
            out_frames, fid = frames_out(cxi_filename, data_shape)  


            dset = fid.create_dataset('entry_1/instrument_1/detector_1/probe', data = probe)
            dset = fid.create_dataset('entry_1/instrument_1/detector_1/data_illumination', data = probe)
            dset = fid.create_dataset('entry_1/instrument_1/source_1/probe', data = probe)
            dset = fid.create_dataset('entry_1/instrument_1/source_1/data_illumination', data = probe)
            dset = fid.create_dataset('entry_1/instrument_1/source_1/illumination', data = probe)
            dset = fid.create_dataset('entry_1/instrument_1/detector_1/probe_mask', data = pMask)

            out_frames[:, :, :] = frames_gather[:, :, :]

            fid.close()

        if nexus_file:

            nexus_filename = output_filename + ".nex"

            printv(color("\nSaving nexus file: " + nexus_filename + "\n", bcolors.OKGREEN))

            #This deletes and rewrites a previous file with the same name
            try:
                os.remove(nexus_filename)
            except OSError:
                pass

            print(metadata)

            metadata["x_translations"] = metadata["translations"][:,0]
            metadata["y_translations"] = metadata["translations"][:,1]
            metadata["z_translations"] = metadata["translations"][:,2]

            data_format = nexus_data
            #writing data
            write(nexus_filename, {"data": frames_gather}, data_format = nexus_data)
            #writing metadata
            write(nexus_filename, metadata, data_format = {**nexus_metadata, **cosmic_metadata})


        
