import os
import re
from astropy.io import fits
import scipy
import scipy.signal

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d

include_path = '/home/simon/common/python/include/'
sys.path.append(include_path)

import astropy.constants as const

c_kms = 1.e-3 * const.c.value  ## light speed in km/s

#import Vtools


def addimage(iplotpos,
             label,
             atitle,
             hdugrey,
             filename_contours,
             filename_errormap=False,
             filename_fiterrormap=False,
             VisibleXaxis=False,
             VisibleYaxis=True,
             DoBeamEllipse=False,
             DoGreyCont=False,
             vsyst=0.,
             nplotsx=2,
             nplotsy=2,
             SymmetricRange=False,
             MedianvalRange=False,
             DoCB=True,
             cmap='RdBu_r',
             MedRms=True,
             Zoom=True,
             PassRange=False):

    print("nplotsx ", nplotsx, nplotsy, iplotpos)
    ax = plt.subplot(nplotsy, nplotsx, iplotpos)
    # ax=axes[iplotpos]

    #ax.xaxis.set_visible(VisibleXaxis)
    #ax.yaxis.set_visible(VisibleYaxis)
    plt.setp(ax.get_xticklabels(), visible=VisibleXaxis)
    plt.setp(ax.get_yticklabels(), visible=VisibleYaxis)

    ax.tick_params(axis='both',
                   length=5,
                   width=1.,
                   color='grey',
                   direction='in',
                   left=True,
                   right=True,
                   bottom=True,
                   top=True)

    ax.spines['right'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['bottom'].set_color('grey')

    if ((iplotpos % nplotsx) == 1):
        ax.set_ylabel(r'$\delta$  offset / arcsec')
    if (iplotpos > (nplotsx * (nplotsy - 1))):
        ax.set_xlabel(r'$\alpha$ offset / arcsec')

    im_grey = hdugrey.data
    hdr_grey = hdugrey.header

    cdelt = 3600. * hdr_grey['CDELT2']

    side0 = hdr_grey['NAXIS2'] * cdelt

    if Zoom:
        side = 3.
        if (side > side0):
            sys.exit("side too large")

        nx = np.rint(side / cdelt)
        ny = np.rint(side / cdelt)

        i_star = np.rint(((0.) / hdr_grey['CDELT1']) +
                         (hdr_grey['CRPIX1'] - 1.))
        j_star = np.rint(((0.) / hdr_grey['CDELT2']) +
                         (hdr_grey['CRPIX2'] - 1.))

        j0 = int(j_star - (ny - 1.) / 2. + 1)
        j1 = int(j_star + (ny - 1.) / 2. + 1)
        i0 = int(i_star - (nx - 1.) / 2. + 1)
        i1 = int(i_star + (nx - 1.) / 2. + 1)
        subim_grey = im_grey[j0:j1, i0:i1]

    else:
        side = side0
        i0 = 0
        i1 = hdr_grey['NAXIS1'] - 1
        j0 = 0
        j1 = hdr_grey['NAXIS2'] - 1

        subim_grey = im_grey.copy()

    a0 = side / 2.
    a1 = -side / 2.
    d0 = -side / 2.
    d1 = side / 2.

    # if 'v' in filename_grey:
    #	subim_grey = subim_grey - vsyst

    print("loading filename_grey", filename_fiterrormap)

    f = fits.open(filename_fiterrormap)
    im_fiterrormap = f[0].data
    hdr_fiterrormap = f[0].header
    if Zoom:
        subim_fiterrormap = im_fiterrormap[j0:j1, i0:i1]
    else:
        subim_fiterrormap = im_fiterrormap.copy()

    # medsubim_fiterrormap=medfilt2d(subim_fiterrormap,kernel_size=11)

    #Vtools.View(subim_fiterrormap)
    #Vtools.View(medsubim_fiterrormap)

    typicalerror = np.median(subim_fiterrormap)

    #mask=np.where((subim_fiterrormap-medsubim_fiterrormap) > 3.* medsubim_fiterrormap)
    mask = np.where(subim_fiterrormap < 3. * typicalerror)

    immask = np.zeros(subim_fiterrormap.shape)
    immask[mask] = 0.
    print("number of pixels masked:", np.sum(immask))
    #print("viewing fiterrormap immask")
    #Vtools.View(immask)

    #plt.figure(2)
    #plt.imshow(immask)
    #plt.show()
    #plt.figure(1)

    if filename_errormap:
        f = fits.open(filename_errormap)
        im_errormap = f[0].data
        hdr_errormap = f[0].header

        if Zoom:
            subim_errormap = im_errormap[j0:j1, i0:i1]
        else:
            subim_errormap = im_errormap.copy()

        medsubim_errormap = medfilt2d(subim_errormap, kernel_size=5)
        mask = np.where(
            mask and
            ((subim_errormap - medsubim_errormap) > 3. * medsubim_errormap))

        immask = np.zeros(subim_fiterrormap.shape)
        immask[mask] = 1.
        print("viewing errormap immask")
        #Vtools.View(immask)

    if SymmetricRange:
        range2 = vsyst + SymmetricRange
        range1 = vsyst - SymmetricRange
        # subim_grey[np.where(subim_grey < range1)]=vsyst
        # subim_grey[np.where(subim_grey > range2)]=vsyst
        clevs = [range1, 0., range2]
        clabels = ['%.0f' % (clevs[0]), '', '%.0f' % (clevs[2])]
    elif MedianvalRange:
        typicalvalue = np.median(subim_grey[mask])
        rms = np.std(subim_grey[mask])
        medrms = np.sqrt(np.median((subim_grey[mask] - typicalvalue)**2))

        print("typical value ", typicalvalue, " rms ", rms, "medrms", medrms)
        range1 = np.min(subim_grey[mask])
        if MedRms:
            imagerms = medrms
        else:
            imagerms = rms
        range2 = typicalvalue + 3. * imagerms
        clevs = [range1, range2]
        clabels = ['%.1f' % (clevs[0]), '%.1f' % (clevs[1])]
    elif PassRange:
        range2 = PassRange[1]
        range1 = PassRange[0]
        clevs = [range1, range2]
        clabels = ['%.1f' % (clevs[0]), '%.1f' % (clevs[1])]
    else:
        range2 = np.max(subim_grey[mask])
        range1 = np.min(subim_grey[mask])
        clevs = [range1, range2]
        clabels = ['%.1f' % (clevs[0]), '%.1f' % (clevs[1])]

    print("max:", np.max(subim_grey))
    print("min:", np.min(subim_grey))
    print("range1", range1, "range2", range2)
    if (np.isnan(subim_grey).any()):
        print("NaNs in subim_grey")
    subim_grey = np.nan_to_num(subim_grey)

    ax.imshow(
        subim_grey,
        origin='lower',
        cmap=cmap,  #norm=norm,
        extent=[a0, a1, d0, d1],
        vmin=range1,
        vmax=range2,
        interpolation='nearest')  #'nearest'  'bicubic'

    #plt.plot(0.,0.,marker='*',color='yellow',markersize=0.2,markeredgecolor='black')
    #plt.plot(0.,0.,marker='*',color='yellow',markersize=1.)

    ax.text(a1 * 0.9,
            d0 * 0.9,
            atitle,
            weight='bold',
            fontsize=12,
            ha='right',
            bbox=dict(facecolor='white', alpha=0.8))

    ax.text(a0 * 0.9,
            d1 * 0.8,
            label,
            weight='bold',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

    axcb = plt.gca()

    if (DoCB):

        cmap1 = cmap
        norm = mpl.colors.Normalize(vmin=range1, vmax=range2)
        fig = plt.gcf()
        cbar_ax = fig.add_axes([0.92, 0.62, 0.01, 0.15])
        if (iplotpos > 2):
            cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.15])

        print("color bar clevs", clevs)
        print("color bar clabels", clabels)

        cb = mpl.colorbar.ColorbarBase(cbar_ax,
                                       cmap=colors.Colormap(cmap1),
                                       norm=norm,
                                       orientation='vertical',
                                       ticks=clevs)

        cb.ax.set_yticklabels(clabels)
        # cb.ax.set_ylabel('km/s', rotation=270)
        cb.ax.tick_params(labelsize=12)

# if (isinstance(DoGreyCont,str)):
#         filename_grey_cont=DoGreyCont
    #
    #
    #
    #         f_cont = fits.open(filename_grey_cont)
    #         im_grey_cont = f_cont[0].data
    #         hdr_grey_cont= f_cont[0].header
    #         subim_grey_cont = im_grey_cont[int(j0):int(j1),int(i0):int(i1)]
    #         print( "i0 "+str(i0),"hdr_grey['CRPIX2']", hdr_grey['CRPIX2'])
    #         levels = [-1.65344452058093, -1.6129840879707 , -1.55158301988206, -1.51002707264227   ]
    #
    #
    #         CS = axcb.contour(subim_grey_cont,levels , origin='lower', linewidths=1.0,
    #                           linestyles = 'solid',
    #                           extent=[a0,a1,d0,d1], colors='red')
    #
    # elif (DoGreyCont):
    #        levels=np.array((vsyst))
    #        CS = axcb.contour(subim_grey,levels , origin='lower', linewidths=0.5,
    #                          linestyles = 'solid',
    #                          extent=[a0,a1,d0,d1], colors='green')

    #if (filename_contours!=False):
    #
    #        ######################################################################
    #        #  contours
    #
    #	print( "loading filename_contours",filename_contours)
    #        f = fits.open(filename_contours)
    #        im_cont = f[0].data
    #        hdr_cont= f[0].header
    #
    #        subim_cont = im_cont[int(j0):int(j1),int(i0):int(i1)]
    #
    #        if MaskCont:
    #                subim_cont[np.where(subim_region < 0.5)] = 0.
    #
    #
    #
    #        ######################################################################
    #        #Blue
    #        print( "full dvel range: ", np.min(subim_cont), " --> ",np.max(subim_cont))
    #        levels=np.array([0.6,])*np.fabs(np.min(subim_cont))
    #        #levels=np.array([0.7421825906386711,])
    #        alphas=np.ones(len(levels)) #  0.5+0.5*np.arange(len(levels))/(len(levels))
    #        linewidths=np.ones(len(levels)) # 1.0-((0.+np.arange(len(levels))-1.0)/(len(levels))) *0.5
    #
    #        levels_list = levels.tolist()
    #        for ilevel in range(len(levels)):
    #                alpha_val = alphas[ilevel]
    #                alinew=linewidths[ilevel]
    #                alevel = -levels[ilevel]
    #                print( "blue level",alevel,"alpha_val",alpha_val,"lw",alinew)
    #                CS = axcb.contour(subim_cont, alevel, origin='lower', linewidths=alinew,
    #                                  linestyles = 'solid',
    #                                  extent=[a0,a1,d0,d1], colors='blue', alpha = alpha_val  )
    #
    #        ######################################################################
    #        #Red
    #        #levels=np.array([0.75,0.95])
    #
    #
    #        levels=np.array([0.6,])*np.fabs(np.max(subim_cont))
    #        #levels=np.array([0.6441364043826032,])
    #        levels_list = levels.tolist()
    #        for ilevel in range(len(levels)):
    #                alpha_val = alphas[ilevel]
    #                alinew=linewidths[ilevel]
    #                alevel = levels[ilevel]
    #                print( "red level",alevel,"alpha_val",alpha_val,"lw",alinew)
    #                CS = axcb.contour(subim_cont, alevel, origin='lower', linewidths=alinew,
    #                                  linestyles = 'solid',
    #                                  extent=[a0,a1,d0,d1], colors='red', alpha = alpha_val  )
    #
    #
    #

    if DoBeamEllipse:
        from matplotlib.patches import Ellipse

        #Bmax/2 0.0579669470623286; Bmin/2 0.038567442164739;
        #PA-51.682370436407deg (South of East);

        bmaj = hdr_grey['BMAJ'] * 3600.
        bmin = hdr_grey['BMIN'] * 3600.
        bpa = hdr_grey['BPA']
        e = Ellipse(xy=[a1 * 0.8, d0 * 0.8],
                    width=bmin,
                    height=bmaj,
                    angle=-bpa,
                    color='blue')
        e.set_clip_box(axcb.bbox)
        e.set_facecolor('yellow')
        e.set_alpha(0.5)
        axcb.add_artist(e)

    return clevs, clabels


def exec_1cube(iplotpos,
               workdir,
               filename_cube,
               vsyst=0.,
               vrange=10.,
               ngauss=2,
               PassRestFreq=-1,
               nchans=5,
               nplotsx=5,
               nplotsy=4,
               cubelabel=''):

    datacube = fits.open(filename_cube)[0].data
    datahdr = fits.open(filename_cube)[0].header
    print("datacube.shape", datacube.shape)

    dnu = datahdr['CDELT3']
    len_nu = datahdr['NAXIS3']
    nui = datahdr['CRVAL3'] - (datahdr['CRPIX3'] - 1) * dnu
    nuf = nui + (len_nu - 1) * dnu

    nu = np.linspace(nui, nuf, len_nu)

    if (PassRestFreq > 0):
        nu0 = PassRestFreq
    elif ('RESTFREQ' in datahdr):
        nu0 = datahdr['RESTFREQ']
    elif ('RESTFRQ' in datahdr):
        nu0 = datahdr['RESTFRQ']
    else:
        sys.exit("no RESTFREQ in HDR, pass RESTFREQ")

    print("using center FREQ", nu0)
    velocities = c_kms * (nu0 - nu) / nu0
    dvel = -c_kms * dnu / nu0
    delta_vel = vrange / float(nchans - 1)
    delta_chan = int(delta_vel / dvel)
    ichan_vsyst = np.argmin(np.fabs(velocities - vsyst))

    selected_channels = arange(nchans) * delta_chan - (delta_chan *
                                                       (float(nchans) - 1.) /
                                                       2, ) + ichan_vsyst

    filename_fiterrormap = workdir + 'fiterrormap.fits'

    range1 = datacube.min()
    range2 = datacube.max()

    for ichan in selected_channels:
        print("ichan ", ichan)
        ichan = int(ichan)
        print("selected velocity ", velocities[ichan])
        print("selected velocity label", "{0:.2f}".format(velocities[ichan]))

        cmap = 'ocean_r'
        atitle = "{0:.2f}".format(velocities[ichan])
        filename_contours = False
        hdugrey = fits.PrimaryHDU()
        hdugrey.data = datacube[ichan, :, :]
        hdugrey.header = datahdr
        filename_errormap = False
        iplotpos += 1
        label = ''

        VisibleYaxis = False
        if ((iplotpos % nplotsx) == 1):
            VisibleYaxis = True
            label = cubelabel

        VisibleXaxis = False
        if ((iplotpos > nplotsx * (nplotsy - 1))):
            VisibleXaxis = True

        print("adding image nplotsx nplotsy", nplotsx, nplotsy)
        (clevs, clabels) = addimage(iplotpos,
                                    label,
                                    atitle,
                                    hdugrey,
                                    filename_contours=filename_contours,
                                    filename_errormap=filename_errormap,
                                    filename_fiterrormap=filename_fiterrormap,
                                    VisibleXaxis=VisibleXaxis,
                                    VisibleYaxis=VisibleYaxis,
                                    DoBeamEllipse=False,
                                    DoGreyCont=False,
                                    vsyst=vsyst,
                                    nplotsx=nplotsx,
                                    nplotsy=nplotsy,
                                    SymmetricRange=False,
                                    DoCB=False,
                                    cmap=cmap,
                                    PassRange=[range1, range2])

    return iplotpos


def exec_summary(workdir,
                 fileout,
                 vsyst=0.,
                 vrange=10.,
                 ngauss=2,
                 filedatacube='',
                 PassRestFreq=-1,
                 nchans=5):

    # global nplotsx
    # global nplotsy
    # global basename_log

    print("workdir:", workdir)
    #matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='sans-serif')
    #matplotlib.rcParams.update({'font.size': 16})
    font = {'family': 'Arial', 'weight': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    size_marker = 10

    # cmaps = ['magma', 'inferno', 'plasma', 'viridis', 'bone', 'afmhot', 'gist_heat', 'CMRmap', 'gnuplot', 'Blues_r', 'Purples_r', 'ocean', 'hot', 'seismic_r']
    gamma = 1.0

    if (ngauss == 2):
        figsize = (14.9, 12.)
        nplotsx = nchans
        nplotsy = 4
    elif (ngauss == 1):
        figsize = (14.9, 6.)
        nplotsx = chans
        nplotsy = 2

    # (fig0, axes) = plt.subplots(nrows=nplotsy,ncols=nplotsx,figsize=figsize)

    plt.figure(figsize=figsize)
    #axes=axes.flatten()
    #print("axes",help(axes))
    #print((axes.shape))

    iplotpos = 0

    #filename_cube = workdir + 'datacube.fits'
    filename_cube = filedatacube
    print(">>>>> doing cube", filename_cube, "iplotpos", iplotpos)
    iplotpos = exec_1cube(iplotpos,
                          workdir,
                          filename_cube,
                          vsyst=vsyst,
                          vrange=vrange,
                          ngauss=ngauss,
                          PassRestFreq=PassRestFreq,
                          nchans=nchans,
                          nplotsx=nplotsx,
                          nplotsy=nplotsy,
                          cubelabel='obs.')

    filename_cube = workdir + 'modelcube.fits'
    print(">>>>> doing cube", filename_cube, "iplotpos", iplotpos)
    iplotpos = exec_1cube(iplotpos,
                          workdir,
                          filename_cube,
                          vsyst=vsyst,
                          vrange=vrange,
                          ngauss=ngauss,
                          PassRestFreq=PassRestFreq,
                          nchans=nchans,
                          nplotsx=nplotsx,
                          nplotsy=nplotsy,
                          cubelabel='mod.')

    if (ngauss == 2):

        filename_cube = workdir + 'modelcube_g1.fits'
        print(">>>>> doing cube", filename_cube, "iplotpos", iplotpos)
        iplotpos = exec_1cube(iplotpos,
                              workdir,
                              filename_cube,
                              vsyst=vsyst,
                              vrange=vrange,
                              ngauss=ngauss,
                              PassRestFreq=PassRestFreq,
                              nchans=nchans,
                              nplotsx=nplotsx,
                              nplotsy=nplotsy,
                              cubelabel=r'g$_1$')

        filename_cube = workdir + 'modelcube_g2.fits'
        print(">>>>> doing cube", filename_cube, "iplotpos", iplotpos)
        iplotpos = exec_1cube(iplotpos,
                              workdir,
                              filename_cube,
                              vsyst=vsyst,
                              vrange=vrange,
                              ngauss=ngauss,
                              PassRestFreq=PassRestFreq,
                              nchans=nchans,
                              nplotsx=nplotsx,
                              nplotsy=nplotsy,
                              cubelabel=r'g$_2$')

    plt.subplots_adjust(hspace=0.)
    plt.subplots_adjust(wspace=0.)

    print(fileout)
    #plt.tight_layout()

    print("USED VSYST=", vsyst)
    plt.savefig(fileout, bbox_inches='tight', dpi=500)

    #plt.savefig(fileout)

    return


#workdir='dgaussminuit_CommonSigma/'
workdir = '12CO_dgaussmoments_2022/'
fileout = workdir + 'fig_channelmap.pdf'

vsyst = 5.76860412669
exec_summary(workdir, fileout, vsyst=vsyst, vrange=5., ngauss=2, nchans=5, filedatacube='/home/simon/HD100546/C6/lines/guvmem_runs/restored_cube_lS0.001_lL0.0_nogrid_kepmask_dV0500_adjust3_robust1.0_z_Resamp.fits')
