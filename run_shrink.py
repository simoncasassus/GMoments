import sys
import os
import re

include_path='/home/simon/common/python/include/'
sys.path.append(include_path)
import GMoments.ShrinkCanvas as ShrinkCanvas





#ShrinkCanvas.Zoom('/strelka_ssd/simon/HD135344B/guvmem_runs/belka_results/12CO21/smooth_model_cube_lS0.003_lL0.0.fits',zoom_area=0.75)

#filein='/strelka_ssd/simon/HD135344B/guvmem_runs/belka_results/12CO21/restored_cube_lS0.003_lL0.0_robust2.0.fits'
#ShrinkCanvas.Zoom(filein,zoom_area=0.75)

filein =  '/strelka_ssd/simon/HD135344B/red/tclean_contsub_uvtaper_xwide_coarse_HD135344Bbriggs2.0_12CO.fits'
#ShrinkCanvas.Zoom(filein,zoom_area=0.75)
ShrinkCanvas.Zoom(filein,zoom_area=2.0)

    
