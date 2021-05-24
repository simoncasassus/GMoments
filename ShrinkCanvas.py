import scipy as sp
import astropy.io.fits as fits
import numpy as np
from copy import copy, deepcopy
import sys
import os
import re

from pprint import pprint

include_path='/home/simon/common/python/include/'
sys.path.append(include_path)
import ImUtils.Resamp as Resamp
import ImUtils.Cube2Im as Cube2Im


def Zoom(cubefile,zoom_area=-1.,pixscale_factor=1.,Resample=False,fileout=''):
    #zoom_area=1.2 # arcsec, half side 

    hducube=fits.open(cubefile)
    datacube = hducube[0].data
    datahdr = hducube[0].header

    hducubeout=deepcopy(hducube)
    print(cubefile)
    print("datacube.shape",datacube.shape)
    if (fileout==''):
        if (Resample or (pixscale_factor != 1.)) :
            fileout=re.sub('.fits','_z_Resamp.fits',cubefile)
        else:
            fileout=re.sub('.fits','_z.fits',cubefile)
            
    print("fileout: ",fileout)
    
    icenter = int(datahdr['CRPIX1']-1.)
    pixscl=pixscale_factor*3600.*datahdr['CDELT2']
    if (zoom_area > 0.):        
        halfside_pix = int(zoom_area/(pixscl))  
        x_i = icenter - halfside_pix
        y_i = icenter - halfside_pix
        x_f = icenter + halfside_pix + 1
        y_f = icenter + halfside_pix + 1
        side_pix = 2*halfside_pix + 1  # NO NEED TO Resamp WITH ODD NUMBER OF  PIXELS
    else:
        sys.exit("pass zoom_area")

    print( "x_1",x_i,"x_f",x_f)


    headcubeout = deepcopy(datahdr)
    imshape=datacube.shape[1:]    
    headcubeout['CRPIX1']= headcubeout['CRPIX1'] - x_i
    headcubeout['CRPIX2']= headcubeout['CRPIX2'] - y_i

    if (len(datacube.shape) > 3):
        headcubeout.pop('CUNIT4', None)
        headcubeout.pop('CTYPE4', None)
        headcubeout.pop('CRVAL4', None)
        headcubeout.pop('CDELT4', None)
        headcubeout.pop('CRPIX4', None)
        headcubeout.pop('NAXIS4', None)


        datacube=datacube[0,:]

        
    if (Resample or (pixscale_factor != 1.)) :
        headcubeout['NAXIS1']=side_pix
        headcubeout['NAXIS2']=side_pix
        headcubeout['CDELT1']=pixscale_factor*headcubeout['CDELT1']
        headcubeout['CDELT2']=pixscale_factor*headcubeout['CDELT2']

        #hduim0=Cube2Im.slice0(hducube,ReturnHDUList=True,DitchCRVAL3=True)
        #hdrim0=hduim0[0].header

        nfreqs=headcubeout['NAXIS3']

        headimout=deepcopy(headcubeout)

        Cube2Im.trimhead(headimout,DitchCRVAL3=True)
        headimout.pop('NAXIS3', None)
        headimout['NAXIS']=2
        

        cubeout=np.zeros((nfreqs,side_pix,side_pix))
        hduim0=Cube2Im.slice0(hducube,ReturnHDUList=True,DitchCRVAL3=True)
        hdrim0=hduim0[0].header
        #hdrim0=Cube2Im.trimhead(hdrim0,DitchCRVAL3=True)

        #pprint(hdrim0)
        #sys.exit()
        
        for k in list(range(nfreqs)):
            print("k= ",k)
            hduim = fits.PrimaryHDU()
            hduim.data = datacube[k,:]
            hduim.header = hdrim0
            #print(hduim.data.shape)
            #sys.exit()
            resamp=Resamp.gridding(hduim,headimout, fullWCS=False)
            cubeout[k,:]=resamp

    else:
        #if (len(datacube.shape) > 3):
        #    cubeout=datacube[0,:,y_i:y_f,x_i:x_f]
        #else:
        cubeout=datacube[:,y_i:y_f,x_i:x_f]

    

    hducubeout[0].data=cubeout
    hducubeout[0].header=headcubeout
    hducubeout.writeto(fileout,overwrite=True)
    
    
