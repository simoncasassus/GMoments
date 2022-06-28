import os
import re
from astropy.io import fits
import scipy
import scipy.signal
import sys

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d

include_path = '/home/simon/common/python/include/'
sys.path.append(include_path)
from GMoments import SummaryDMoments_compact
from GMoments import SummaryDMoments

workdirs = ['AS209_12CO_dgauss/',]

filename_continuum = False

vsyst = 5.66
for aworkdir in workdirs:
    fileout = aworkdir + 'fig_summary_report.pdf'
    ngauss = 1
    if ('dgauss' in aworkdir):
        ngauss = 2
        print('aworkdir ', aworkdir)
        SummaryDMoments.exec_summary(aworkdir,
                                     fileout,
                                     vsyst=vsyst,
                                     vrange=10.,
                                     ngauss=ngauss,
                                     WCont=False,
                                     Zoom=False,
                                     Side=5.,
                                     filename_continuum=filename_continuum,
                                     contlevels=[2.803e-03 / 10.])
    else:
        SummaryDMoments.exec_summary(aworkdir,
                                     fileout,
                                     vsyst=vsyst,
                                     vrange=10.,
                                     ngauss=ngauss,
                                     WCont=False,
                                     Zoom=False,
                                     Side=3.,
                                     filename_continuum=filename_continuum)
