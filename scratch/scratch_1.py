#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot figures
Figon = False
Figon = True

# pick either final_res or img_width
final_res = 5e-9 # desired final pixel size nanometers
#final_res = None # desired final pixel size nanometers
img_width= 1200 # cropped width of the raw clean frames

frame_pixels=256 # final pixels 
t12 = 5 # time ratio between long and short exposure

# input file
data_dir='/tomodata/NS/200220033/'
h5fname=data_dir+'raw_NS_200220033_026.cxi'
# output file
h5fname_out=data_dir+'filtered_NS_200220033_026.cxi'


import numpy as np
import scipy.constants


import fccd
from fccd import imgXraw as imgXraw

# combine double exposure
def combine(data0, data1, thres=3e3):
    msk=data0<thres
    return (t12+1)*(data0*msk+data1)/(t12*msk+1)


# get background
import h5py

fid = h5py.File(h5fname, 'r')

# characterization of the dark field 
# get the background
bkg=np.array(fid['entry_1/data_1/dark_data'])

# split the average from 2 exposures:
bkg_avg0=np.average(bkg[0::2],axis=0)
bkg_avg1=np.average(bkg[1::2],axis=0)

## get one frame to compute center
rdata = fid['entry_1/data_1/raw_data']
n_frames = rdata.shape[0]//2 # number of frames (1/2 for double exposure)

# split short and long exposures
ii=2500//2+25 # middle frame, we could use the first
rdata0=rdata[ii*2]-bkg_avg0
rdata1=rdata[ii*2+1]-bkg_avg1

# from metadata
# Energy (converted to keV)
# E= fid['entry_1/instrument_1/source_1/energy'][...]*1/scipy.constants.elementary_charge
E = 1300 #eV



# get one clean frame
img0=combine(imgXraw(rdata0),imgXraw(rdata1))

width=img0.shape[0]

# get the width from the desired resolution
if final_res is not None:
    ccd_pixel=fccd.ccd_pixel
    ccd_dist = fid['entry_1/instrument_1/detector_1/distance'][...]
    hc=scipy.constants.Planck*scipy.constants.c/scipy.constants.elementary_charge
    wavelength = hc/E

    img_width= width/(ccd_dist*wavelength/(ccd_pixel*width)/final_res) # cropped width of the raw clean frames
    #img_width = resolution2frame_width(ccd_dist,E,ccd_pixel,heigth,final_res)


import filter_frames
center_of_mass, filter_img, shift_rescale = filter_frames.init(width, frame_pixels, img_width)

# we need a shift, we take it from the first frame:
com = center_of_mass(img0*(img0>0))-width//2
com = np.round(com)

def frameXclean(img):
    return shift_rescale(img,com)


# plot an image

if Figon:
    img2 = filter_img(img0)
    #img3 = rescale(img2)
    img3 = frameXclean(img2)
    #img3 = img3*(img3>0) # positive

    import matplotlib.pyplot as plt
    plt.imshow(img2)
    plt.figure(2)
    plt.imshow(img3)
    plt.draw()



# this is a copy of the raw data
# output file should have all the metadata already

fido = h5py.File(h5fname_out, 'a')
try:
    del fido['entry_1/data_1/dark_data']
except:
    None
try:
    del fido['entry_1/data_1/raw_data']
except:
    None

try:
    del fido['entry_1/data_1/raw_data']
except:
    None

try:
    del fido['entry_1/data_1/data']
except:
    None

try:
    del fido['entry_1/image_1']
    del fido['entry_1/image_2']
except:
    None
try:
    del fido['entry_1/image_latest']
except: None

try:
    del fido['/entry_1/instrument_1/source_1/probe_mask']
except: None
try:
    del fido['/entry_1/instrument_1/source_1/illumination_intensities']
    del fido['/entry_1/instrument_1/source_1/illumination']
except: None
    
    


# modify pixel size. pixel size is rescaled
x_pixel_size=ccd_pixel*img_width/frame_pixels
#####################
try:
    del fido['entry_1/instrument_1/detector_1/x_pixel_size']
    del fido['entry_1/instrument_1/detector_1/y_pixel_size']
except: None
    
fido['entry_1/instrument_1/detector_1/x_pixel_size']=x_pixel_size
fido['entry_1/instrument_1/detector_1/y_pixel_size']=x_pixel_size

fido.flush()

out_data=fido.create_dataset("entry_1/data_1/data", (n_frames, frame_pixels,frame_pixels), dtype='f')


import sys


figure = None
for ii in np.arange(n_frames):

    img0 = combine(imgXraw(rdata[ii*2]-bkg_avg0),imgXraw(rdata[ii*2+1]-bkg_avg1))
    img2 = filter_img(img0)
    img3 = frameXclean(img0)
    
    out_data[ii] = img3
    #print('hello')
    sys.stdout.write('\r frame = %s/%s ' %(ii+1,n_frames))
    sys.stdout.flush()

    if Figon:
        if figure is None:
            figure = plt.imshow((np.abs(img3)+.2)**.2)
            #figure = plt.imshow((np.abs(img3)))
            ttl=plt.title
        else:
            figure.set_data((np.abs(img3)+.2)**.2)
            #figure.set_data((np.abs(img3)))
        ttl('frame'+str(ii))
                
        plt.pause(.01)
        plt.draw()


fido.close()
