import os
import sys
import re
include_path=os.environ['HOME']+'/common/python/include/'
sys.path.append(include_path)
#import  DGaussMinuit 
import  GMoments.DGaussMoments as DGaussMoments
#import  DGaussMoments 




restfreq=2.305380000000E+11

n_cores=20

filein='/strelka_ssd/simon/MAPSKINE/data/AS_209_CO_220GHz.0.3arcsec.image_z.fits'




workdirgaussmoments='AS209_12CO_dgauss/'
os.system("rm -rf "+workdirgaussmoments)
os.system('mkdir '+workdirgaussmoments)
DGaussMoments.exec_Gfit(filein,workdirgaussmoments,wBaseline=False,n_cores=n_cores,zoom_area=-1,Noise=2.7E-3,Clip=False,DoubleGauss=True,StoreModel=False,Randomize2ndGauss=True,ShrinkCanvas=False,UseCommonSigma=False,PassRestFreq=restfreq)

