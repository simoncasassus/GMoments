import scipy as sp
import astropy.io.fits as pf
import numpy as np
from copy import copy, deepcopy
import sys
import os
import re


def Zoom(cubefile,zoom_area=-1.):
    #zoom_area=1.2 # arcsec, half side 

    hducube=pf.open(cubefile)
    datacube = hducube[0].data
    datahdr = hducube[0].header

    hducubeout=deepcopy(hducube)
    
    print("datacube.shape",datacube.shape)
    fileout=re.sub('.fits','_z.fits',cubefile)
    print("fileout: ",fileout)
    
    icenter = int(datahdr['CRPIX1']-1.)     
    if (zoom_area > 0.):        
        halfside_pix = int(zoom_area/(3600.*datahdr['CDELT2']))  
        x_i = icenter - halfside_pix
        y_i = icenter - halfside_pix
        x_f = icenter + halfside_pix + 1
        y_f = icenter + halfside_pix + 1
        side_pix = 2*halfside_pix + 1  # NO NEED TO Resamp WITH ODD NUMBER OF  PIXELS
    else:
        sys.exit("pass zoom_area")

    print( "x_1",x_i,"x_f",x_f)


    headcubeout = deepcopy(datahdr)
    if (len(datacube.shape) > 3):
        cubeout=datacube[0,:,y_i:y_f,x_i:x_f]
        headcubeout.pop('CUNIT4', None)
        headcubeout.pop('CTYPE4', None)
        headcubeout.pop('CRVAL4', None)
        headcubeout.pop('CDELT4', None)
        headcubeout.pop('CRPIX4', None)
    else:
        cubeout=datacube[:,y_i:y_f,x_i:x_f]
        
    imshape=datacube.shape[1:]
    headcubeout['CRPIX1']= headcubeout['CRPIX1'] - x_i
    headcubeout['CRPIX2']= headcubeout['CRPIX2'] - y_i

    hducubeout[0].data=cubeout
    hducubeout[0].header=headcubeout
    hducubeout.writeto(fileout,overwrite=True)
    
    
