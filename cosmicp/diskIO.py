import h5py
import numpy as np
import sys
import os
import json
from PIL import Image
from .common import printd, printv, color, bcolors


def read_metadata_hdf5(fname):

    with h5py.File(fname, "r") as f:
	    meta = json.loads(f["metadata/"][()])
    return meta


def read_metadata(json_file):

    with open(json_file) as f:
        metadata = json.load(f)

    return metadata


def read_dark_data(metadata, json_file):


    dark_frames = None

    base_folder = os.path.split(json_file)[:-1][0] + "/"

    if "dark_dir" in metadata:

        printv(color("\nReading dark frames from disk...\n", bcolors.OKGREEN))

        #by default we could take the full path, but in this case it has an absolute path from PHASIS,
        #which is not good if you move data to other places
        dark_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["dark_dir"])))


    return dark_frames


def read_tiffs(directory, my_indexes = None):
    from cosmicp.common import rank

    lst=os.listdir(directory)
    lst.sort()

    #We remove from the list anything that is not a tiff
    lst = [item for item in lst if item.endswith('.tif')]

    if my_indexes is not None:
        lst = lst[my_indexes]

    frames = []    

    ii=0
    n_frames=len(lst)

    for fname in lst:       
        ii+=1
        im = Image.open(os.path.join(directory, fname))
        imarray = np.array(im)
        frames.append(imarray)
        if rank == 0:
            sys.stdout.write(color("\r File = %s/%s " %(ii,n_frames), bcolors.HEADER))
            sys.stdout.flush()
        
    if rank == 0: print("\n")
    return np.array(frames)


def frames_out(file_name, shape_frames):
    import h5py

    fid = h5py.File(file_name, 'a')
        
    if not "entry_1/data_1/" in fid: 
        fid.create_group("entry_1/data_1")
    if not '/entry_1/instrument_1/detector_1/' in fid:
        fid.create_group('/entry_1/instrument_1/detector_1/')

    #out_frames = fid.create_dataset('entry_1/data_1/data', shape_frames , dtype='float32')
    out_frames = fid.create_dataset('/entry_1/instrument_1/detector_1/data', shape_frames , dtype='float32')
    

    if "entry_1/instrument_1/detector_1/data" in fid and not "entry_1/data_1/data" in fid:
        fid["entry_1/instrument_1/detector_1/data"].attrs['axes'] = "translation:y:x" 
        fid["entry_1/data_1/data"] = h5py.SoftLink("/entry_1/instrument_1/detector_1/data")

            

    return out_frames, fid

def map_tiffs(base_folder):
    import tifffile
    ## tifs = tifffile.TiffSequence(lst)
    ##tifs = tifffile.TiffSequence(base_folder)
    tifs = tifffile.TiffSequence(base_folder+'/*.tif')
    class MyClass():
        def __getitem__(self, key):
            return np.array(tifs.asarray(key))
        shape=tifs.shape

    myobj = MyClass()
    
    return myobj

class IO:

    def __init__(self):

        groups = {
                "tomography": "tomography/",
                "data": "entry_1/data_1/",
                "geometry": "entry_1/sample_1/geometry_1/",
                "source":  "entry_1/instrument_1/source_1/", 
                "detector": "entry_1/instrument_1/detector_1/",
                "process": "entry_1/image_1/process_1/"
            }
 
        self.dataFormat = {"data": [groups["detector"] + "data"], #This contains pre-processed data. It is 3D if experiment is 2D ptychography, 4D if it is ptycho-tomography.
                           "raw_data": [groups["detector"] + "raw_data"], #This would contain un-preprocess raw frames. It could have multiple exposure measurements.
                           "dark_data": [groups["detector"] + "dark_data"] #This contains raw dark frames.
                          }               

        self.metadataFormat = {

                #Tomography experimental fields

                "angles": [groups["tomography"] + "angles"],

                #Ptychography experimental fields 

                "translations": [groups["geometry"] + "translation"], #translations in meters
                "pix_translations": [groups["geometry"] + "pix_translations"], #translations in pixels
                "energy": [groups["source"] + "energy"],
                "illumination": 
                    [groups["source"] + "illumination", 
                    groups["source"] + "probe",
                    groups["detector"] + "probe",
                    groups["detector"] + "data_illumination"],
                "detector_distance": [groups["detector"] + "distance"],
                "illumination_distance": [groups["source"] + "illumination_distance"],
                "x_pixel_size": [groups["detector"] + "x_pixel_size"],
                "y_pixel_size": [groups["detector"] + "y_pixel_size"],
                "illumination_mask":
                    [groups["source"] + "probe_mask",
                     groups["detector"] + "probe_mask"],

                "illumination_intensities":
                    [groups["source"] + "illumination_intensities",
                     groups["detector"] + "illumination_intensities"],

                "near_field": [groups["detector"] + "near_field"],
                "pinhole_width": [groups["source"] + "pinhole_width"],
                "phase_curvature": [groups["source"] + "phase_curvature"],

                #Ptychography reconstruction fields
                "final_illumination": [groups["process"] + "final_illumination"],
                "image_x": [groups["process"] + "image_x"],
                "image_y": [groups["process"] + "image_y"],
                "reciprocal_res": [groups["process"] + "reciprocal_resolution"],

                "detector_mask": [groups["detector"] + "detector_mask"],
                "reflection_angles": [groups["detector"] + "reflection_angles"],

                #This is relevant only with raw un-preprocessed data
                "n_exposures": [groups["detector"] + "number_exposures"],

                #This is needed in SHARP although it is not used
                "corner_position": [groups["detector"] + "corner_position"]

            }


    def read(self, file_name, data_format = None, data_indexes = ()):

        if data_format is None: 
            data_format = self.metadataFormat 

        data_dictionary = {}
        try:
            with h5py.File(file_name, "r") as f:
                for key, value in data_format.items():
                    for i in value: #specific data can be stored in different fields of a format, this loop handles that 
                        if i in f : #if group exists...
                            data = f[i][data_indexes]
                            data_dictionary.update({key : data})
                            break
            return data_dictionary
        except IOError:
            printv("***ERROR*** opening file: " + file_name)
            return 0 
           
    #This is used by the new preprocessor prototype, can be removed in the future   
    def write(self, file_name, data_dictionary, data_format = None):

        if data_format is None: 
            data_format = self.metadataFormat 

        with h5py.File(file_name, 'a') as f:
            
            #This below is needed by SHARP
            if not "cxi_version" in f: 
                f.create_dataset("cxi_version",data=140)

            if not "entry_1/data_1/" in f: 
                f.create_group("entry_1/data_1")

            
            #data_format is used to generate hdf5 fields 
            for key, value in data_dictionary.items():
                #we may have here fields that are not considered in the formats above, we still handle them
                try:
                    group = data_format[key][0]
                except KeyError:
                    group = key + "/"

                if value is not None:
                    f.create_dataset(group, data = value)

            #This below is needed by SHARP
            if "entry_1/instrument_1/detector_1/data" in f and not "entry_1/data_1/data" in f:
                f["entry_1/instrument_1/detector_1/data"].attrs['axes'] = "translation:y:x" 
                f["entry_1/data_1/data"] = h5py.SoftLink("/entry_1/instrument_1/detector_1/data")

            if "entry_1/sample_1/geometry_1/translation" in f and not "entry_1/data_1/translation" in f and not "entry_1/instrument_1/detector_1/translation" in f:
                f["entry_1/data_1/translation"] = h5py.SoftLink("/entry_1/sample_1/geometry_1/translation")
                f["entry_1/instrument_1/detector_1/translation"] = h5py.SoftLink("/entry_1/sample_1/geometry_1/translation")

 
