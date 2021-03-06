import scipy as sp
import astropy.io.fits as pf
from scipy.integrate import simps
import numpy as np
from astropy import constants as const
from copy import copy, deepcopy
import sys
import os
import astropy.units as u
import astropy.constants as const
from iminuit import Minuit
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import re
from operator import itemgetter, attrgetter


if not sys.warnoptions:
    import os, warnings
    #warnings.simplefilter("default") # Change the filter in this process
    warnings.simplefilter("ignore") # Change the filter in this process
    #os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

c_kms = 1.e-3 * const.c.value  ## light speed in km/s

def gaussian(x,a,mu,sigma):
    return a * sp.exp(-.5 * ( (x - mu) / sigma )**2.)


def neggaussfit(x,a,mu,sigma,a2,mu2,sigma2):
    model= gaussian(x,a,mu,sigma) + gaussian(x,a2,mu2,sigma2)
    return -model


def v2(x,a,mu,sigma,a2,mu2,sigma2):
    print("in v2")
    return x**2


def chi2_2gauss_wbase_commonsigma(x,a,mu,sigma,a2,mu2,obsspectrum,rmsnoise,baseparams=[0.,0.]):
    baseline=baseparams[0]*x+baseparams[1]
    model= baseline+gaussian(x,a,mu,sigma) + gaussian(x,a2,mu2,sigma)
    aux = (np.abs(obsspectrum-model))**2.
    chi2 = sp.sum(aux)
    chi2 = chi2/rmsnoise**2.
    return chi2

def chi2_2gauss_wbase(x,a,mu,sigma,a2,mu2,sigma2,obsspectrum,rmsnoise,baseparams=[0.,0.]):
    baseline=baseparams[0]*x+baseparams[1]
    model= baseline+gaussian(x,a,mu,sigma) + gaussian(x,a2,mu2,sigma2)
    aux = (np.abs(obsspectrum-model))**2.
    chi2 = sp.sum(aux)
    chi2 = chi2/rmsnoise**2.
    return chi2


def chi2_2gauss(x,a,mu,sigma,a2,mu2,sigma2,obsspectrum,rmsnoise):
    model= gaussian(x,a,mu,sigma) + gaussian(x,a2,mu2,sigma2)
    aux = (np.abs(obsspectrum-model))**2.
    chi2 = sp.sum(aux)
    chi2 = chi2/rmsnoise**2.
    return chi2

def chi2_gauss_wbase(x,a,mu,sigma,obsspectrum,rmsnoise,baseparams=[0.,0.]):
    baseline=baseparams[0]*x+baseparams[1]
    gauss = baseline+gaussian(x,a,mu,sigma)
    aux = (np.abs(obsspectrum-gauss))**2.
    chi2 = sp.sum(aux)
    chi2 = chi2/rmsnoise**2.
    return chi2

def chi2_gauss(x,a,mu,sigma,obsspectrum,rmsnoise):
    gauss = gaussian(x,a,mu,sigma)
    aux = (np.abs(obsspectrum-gauss))**2.
    chi2 = sp.sum(aux)
    chi2 = chi2/rmsnoise**2.
    return chi2



def fitter(n):
    #if ( ((float(npix)/float(n))-int(float(npix)/float(n))) == 0):
    #    print( (float(npix)/float(n)),"%")

    i = int(n/side_pix)
    j = int(n%side_pix)
    i += x_i
    j += y_i
    #example near center with signal
    #i=1105
    #j=1061
    #print( "i",i,"j",j,"x_i",x_i)

    if len(cube.shape)>3:
        signal = cube[0,:,j,i]
    else:
        signal = cube[:,j,i]


    if (DoClip):
        signal[np.where(signal < 0)] = 0.
    if (signal.max() <= 0):
        return [None]
    mask=signal>-0.01*signal.max() ##mask for values too negative.
    selected_velocities = velocities[mask]      
    signal_a=signal[mask]
    Amp_init=signal_a.max() - signal_a.mean()
    if (Amp_init  <= 0):
        return [None]
    #v0_init = selected_velocities[signal_a==signal_a.max()]
    v0_init = selected_velocities[np.argmax(signal_a)] # selected_velocities[signal_a==signal_a.max()]
    sigma_init=0.1
    sigma_max=30.
    ## take the error as the rmsnoise far from the line
    #noise = signal[(velocities<v0_init-1.) | (velocities>v0_init+1.)]
    #rmsnoise = sp.sqrt(sp.mean(noise**2.))
    FixBase_a=True
    FixBase_b=True
    if (DoBaseline):
        FixBase_a=False
        FixBase_b=False
    if (DGauss):
        # velorange=velocities.min()-1.0, velocities.max()+1.0
        v0_init2 = v0_init
        if (Randomize):
            v0_init2 += 3.*dv*(np.random.random()-0.5) # selected_velocities[signal_a==signal_a.max()]

        maxslope=Amp_init/np.fabs((np.max(selected_velocities)-np.min(selected_velocities)))
        if (CommonSigma):
            f = lambda a,mu,sigma,a2,mu2,sigma2,base_a,base_b: chi2_2gauss_wbase_commonsigma(selected_velocities, a, mu, sigma, a2, mu2, signal_a, rmsnoise, baseparams=[base_a,base_b])
            m = Minuit(f, a=Amp_init, mu=v0_init, sigma=sigma_init, a2=Amp_init, mu2=v0_init2, base_a=0.,base_b=0.,#initial guess
                       errordef=1,                      # error
                       error_a=Amp_init*0.001, #stepsizes for pars
                       error_mu=.1,
                       error_sigma=.1*sigma_init,
                       error_a2=Amp_init*0.001, #stepsizes for pars
                       error_mu2=.1,
                       error_base_a=maxslope*0.001,
                       error_base_b=Amp_init*0.001,
                       limit_a=(0., 5.*Amp_init), #Bounds for pars
                       limit_mu=(velocities.min()-1.0, velocities.max()+1.0),
                       limit_sigma=(dv/2., sigma_max),
                       limit_a2=(0., 5.*Amp_init), #Bounds for pars
                       limit_mu2=(velocities.min()-1.0, velocities.max()+1.0),
                       limit_base_a=(-maxslope,maxslope),
                       limit_base_b=(0.,signal_a.max()),
                       print_level=0,
                       pedantic=False,
                       fix_base_a = FixBase_a,
                       fix_base_b = FixBase_b
                       )
        else:
            f = lambda a,mu,sigma,a2,mu2,sigma2,base_a,base_b: chi2_2gauss_wbase(selected_velocities, a, mu, sigma, a2, mu2, sigma2, signal_a, rmsnoise, baseparams=[base_a,base_b])
            m = Minuit(f, a=Amp_init, mu=v0_init, sigma=sigma_init, a2=Amp_init, mu2=v0_init2, sigma2=sigma_init, base_a=0.,base_b=0.,#initial guess
                       errordef=1,                      # error
                       error_a=Amp_init*0.001, #stepsizes for pars
                       error_mu=.1,
                       error_sigma=.1*sigma_init,
                       error_a2=Amp_init*0.001, #stepsizes for pars
                       error_mu2=.1,
                       error_sigma2=.1*sigma_init,
                       error_base_a=maxslope*0.001,
                       error_base_b=Amp_init*0.001,
                       limit_a=(0., 5.*Amp_init), #Bounds for pars
                       limit_mu=(velocities.min()-1.0, velocities.max()+1.0),
                       limit_sigma=(dv/2., sigma_max),
                       limit_a2=(0., 5.*Amp_init), #Bounds for pars
                       limit_mu2=(velocities.min()-1.0, velocities.max()+1.0),
                       limit_sigma2=(dv/2., sigma_max),
                       limit_base_a=(-maxslope,maxslope),
                       limit_base_b=(0.,signal_a.max()),
                       print_level=0,
                       pedantic=False,
                       fix_base_a = FixBase_a,
                       fix_base_b = FixBase_b
                       )
    else:
        maxslope=Amp_init/np.fabs((np.max(selected_velocities)-np.min(selected_velocities)))
        f = lambda a,mu,sigma,base_a,base_b: chi2_gauss_wbase(selected_velocities, a, mu, sigma, signal_a, rmsnoise, baseparams=[base_a,base_b])
        m = Minuit(f, a=Amp_init, mu=v0_init, sigma=sigma_init, base_a=0.,base_b=0.,#initial guess
                   errordef=1,                      # error
                   error_a=Amp_init*0.001, #stepsizes for pars
                   error_mu=.1,
                   error_sigma=.1*sigma_init,
                   error_base_a=maxslope*0.001,
                   error_base_b=Amp_init*0.001,
                   limit_a=(0., 5.*Amp_init), #Bounds for pars
                   limit_mu=(velocities.min()-1.0, velocities.max()+1.0),
                   limit_sigma=(dv/2., sigma_max),
                   limit_base_a=(-maxslope,maxslope),
                   limit_base_b=(0.,signal_a.max()),
                   print_level=0,
                   pedantic=False,
                   fix_base_a = FixBase_a,
                   fix_base_b = FixBase_b
                   )

    m.migrad()
    m.hesse() 


    if (DGauss):
        if CommonSigma:
            pars = [m.values['a'], m.values['mu'], m.values['sigma'],m.values['a2'], m.values['mu2'], m.values['sigma']]   # pars for best fit
            err_pars = [m.errors['a'], m.errors['mu'], m.errors['sigma'],m.errors['a2'], m.errors['mu2'], m.errors['sigma']]  #error in pars
        else:
            pars = [m.values['a'], m.values['mu'], m.values['sigma'],m.values['a2'], m.values['mu2'], m.values['sigma2']]   # pars for best fit
            err_pars = [m.errors['a'], m.errors['mu'], m.errors['sigma'],m.errors['a2'], m.errors['mu2'], m.errors['sigma2']]  #error in pars

        amps=[ [pars[0],0],[pars[3],3]]
        amps_sorted=sorted(amps,key=itemgetter(0))
        i_G1=amps_sorted[-1][1]
        i_G2=amps_sorted[0][1]

        g_amp=pars[i_G1]
        g_v0=pars[i_G1+1]
        g_sigma=pars[i_G1+2]
        g_amp_e=err_pars[i_G1]
        g_v0_e = err_pars[i_G1+1]  
        g_sigma_e = err_pars[i_G1+2]  
        gaussfit1=gaussian(velocities, g_amp,g_v0,g_sigma)
        
        g2_amp=pars[i_G2]
        g2_v0=pars[i_G2+1]
        g2_sigma=pars[i_G2+2]
        g2_amp_e=err_pars[i_G2]
        g2_v0_e=err_pars[i_G2+1]
        g2_sigma_e=err_pars[i_G2+2]
        gaussfit2=gaussian(velocities, g2_amp,g2_v0,g2_sigma)

        fit1=gaussfit1+gaussfit2


        vpeak=velocities[np.argmax(fit1)]
        ComputeG8=True
        if ComputeG8:
            vpeak_init=vpeak


            f_vpeak = lambda vmax: neggaussfit(vmax, g_amp, g_v0, g_sigma, g2_amp, g2_v0, g2_sigma)
            mvpeak = Minuit(f_vpeak, vmax=vpeak_init,  #initial guess
                            errordef=1,                      # error
                            error_vmax=dv*0.01,
                            #limit_vmax=(velocities.min()-1.0, velocities.max()+1.0),
                            limit_vmax=(np.min(selected_velocities), np.max(selected_velocities)), #Bounds for pars
                            print_level=0,
                            )
            mvpeak.migrad()
            
            vpeak=mvpeak.values['vmax']
            #if ( (i==20) and  (j>10) and (j<30)):
            #    print("mvpeak",mvpeak.values)
            #    print (g_amp, g_v0, g_sigma, g2_amp, g2_v0, g2_sigma)
            #    print( "vpeak",vpeak,"vpeak_init",vpeak_init)
            #    print( "dv",dv)

    else:
        g_amp=m.values['a']
        g_v0=m.values['mu']
        g_sigma=m.values['sigma']
        g_amp_e=m.errors['a']
        g_v0_e=m.errors['mu']
        g_sigma_e=m.errors['sigma']

        gaussfit1=gaussian(velocities, g_amp,g_v0,g_sigma)
        fit1=gaussfit1

    if (DoBaseline):
        base_a = m.values['base_a']
        base_b = m.values['base_b']
        baseline=base_a*velocities+base_b
        fit1 += baseline
            

    fiterror = np.std(fit1-signal)
    
    #fitinit = gaussian(velocities, Amp_init,v0_init,sigma_init)
    #plt.plot(velocities, signal)
    #plt.plot(velocities, fit1)
    #plt.plot(velocities, fitinit)
    #plt.show()

    gmom_0 = abs(simps(fit1,velocities))


  
    # print( "vel range:",np.min(velocities),np.max(velocities),"best fit",pars[1])
    ic=np.argmin(abs(velocities-g_v0))
    if (g_sigma < sigma_init):
        Delta_i = sigma_init/dv
    else:
        Delta_i = g_sigma/dv

    nsigrange=5
    i0=int(ic-nsigrange*Delta_i)
    if (i0 < 0):
        i0=0
    i1=int(ic+nsigrange*Delta_i)
    if (i1 > (len(velocities)-1)):
        i1=(len(velocities)-1)
    j0=int(ic-nsigrange*Delta_i)
    if (j0 < 0):
        j0=0        
    j1=int(ic+nsigrange*Delta_i)
    if (j1 > (len(velocities)-1)):
        j1=(len(velocities)-1)

    #print( "i0",i0,"i1",i1,"j0",j0,"j1",j1)

    sign=1.
    if (velocities[1]<velocities[0]):
        sign=-1

    Smom_0 = sign*simps(signal[i0:i1],velocities[i0:i1])
    Smom_1 = simps(signal[i0:i1]*velocities[i0:i1],velocities[i0:i1])
    subvelo=velocities[i0:i1]
    Smom_8 = subvelo[np.argmax(signal[i0:i1])]
    Smax = np.max(signal[i0:i1])
    
    if (abs(Smom_0) > 0):
        Smom_1 /= Smom_0
        if (Smom_0 > 0):
            var = sign*simps(signal[i0:i1]*(velocities[i0:i1] - Smom_1)**2,velocities[i0:i1])
            if (var > 0):
                Smom_2=np.sqrt(var/Smom_0)
            else:
                Smom_2=-1E6
        else:
                Smom_2=-1E6
    else:
        Smom_1 = -1E6
        Smom_2 = -1E6


    sol = [i,j,gmom_0,g_amp,g_amp_e,g_v0,g_v0_e,g_sigma,g_sigma_e,Smom_0,Smom_1,Smom_2,Smom_8,fiterror,gaussfit1]
    if DGauss:
        sol.extend([gaussfit2,g2_amp,g2_amp_e,g2_v0,g2_v0_e,g2_sigma,g2_sigma_e,vpeak])
    if (DoBaseline): 
        sol.extend([base_a,base_b,baseline])
    sol.extend([Smax,])

    return sol


def exec_Gfit(cubefile,workdir,wBaseline=False,n_cores=30,zoom_area=-1.,Noise=1.0,Clip=False,DoubleGauss=False,StoreModel=False,Randomize2ndGauss=True,ShrinkCanvas=True,UseCommonSigma=False,PassRestFreq=-1):
    #Region=True: zoom into central region, defined as nx/2., with half side zoom_area 
    #zoom_area=1.2 # arcsec

    global n
    global side_pix
    global cube 
    global x_i
    global y_i
    global velocities
    global dv
    global DoBaseline
    global rmsnoise
    global DoClip
    global DGauss
    global Randomize
    global CommonSigma

    DoClip=Clip
    rmsnoise=Noise
    DoBaseline=wBaseline
    DGauss=DoubleGauss
    Randomize=Randomize2ndGauss
    CommonSigma=UseCommonSigma

    datacube = pf.open(cubefile)[0].data
    datahdr = pf.open(cubefile)[0].header

    print("datacube.shape",datacube.shape)

    # if len(cube.shape)>3:
    #    # cube = cube[1:]
    #    cube = cube[0,:,:,:]
    # print("cube.shape",cube.shape)

    # cube = sp.swapaxes(cube,0,2)
    # cube = sp.swapaxes(cube,0,1)

    dnu = datahdr['CDELT3']
    len_nu = datahdr['NAXIS3']
    nui = datahdr['CRVAL3']- (datahdr['CRPIX3']-1)*dnu
    nuf = nui + (len_nu-1)*dnu
    
    nu = sp.linspace(nui, nuf, len_nu)

    if (PassRestFreq>0):
        nu0=PassRestFreq
    elif ('RESTFREQ' in datahdr):
        nu0 = datahdr['RESTFREQ']
    elif ('RESTFRQ' in datahdr):
        nu0 = datahdr['RESTFRQ']
    else:
        sys.exit("no RESTFREQ in HDR, pass RESTFREQ")


    print("using center FREQ", nu0)
    velocities = c_kms*(nu0-nu)/nu0


    icenter = int(datahdr['CRPIX1']-1.)     
    if (zoom_area > 0.):        
        halfside_pix = int(zoom_area/(3600.*datahdr['CDELT2']))  
        x_i = icenter - halfside_pix
        y_i = icenter - halfside_pix
        x_f = icenter + halfside_pix + 1
        y_f = icenter + halfside_pix + 1
        side_pix = 2*halfside_pix + 1
    else: 
        x_i = 0 
        y_i = 0
        x_f = datahdr['NAXIS1']-1+1
        y_f = datahdr['NAXIS1']-1+1
        side_pix = datahdr['NAXIS1'] 

    print( "x_1",x_i,"x_f",x_f)

    
    npix=int(side_pix**2)
    print( "npix",npix)


    headcube = deepcopy(datahdr)
    if ShrinkCanvas:
        if (len(datacube.shape) > 3):
            cube=datacube[0,:,y_i:y_f,x_i:x_f]
            headcube.pop('CUNIT4', None)
            headcube.pop('CTYPE4', None)
            headcube.pop('CRVAL4', None)
            headcube.pop('CDELT4', None)
            headcube.pop('CRPIX4', None)
        else:
            cube=datacube[:,y_i:y_f,x_i:x_f]

        imshape=cube.shape[1:]
        headcube['CRPIX1']= headcube['CRPIX1'] - x_i
        headcube['CRPIX2']= headcube['CRPIX2'] - y_i

        x_i=0
        y_i=0
        x_f=side_pix
    else:
        cube=datacube
        if (len(datacube.shape) > 3):
            imshape=cube.shape[2:]
        else:
            imshape=cube.shape[1:]
    

    im_gmom_0 = sp.zeros(imshape)

    im_g_a = sp.zeros(imshape)
    im_g_v0 = sp.zeros(imshape)
    im_g_sigma = sp.zeros(imshape)
    im_g_a_e = sp.zeros(imshape)
    im_g_v0_e = sp.zeros(imshape)
    im_g_sigma_e = sp.zeros(imshape)
    im_gmom_8 = sp.zeros(imshape)

    SSmom_0 = sp.zeros(imshape)
    SSmom_1 = sp.zeros(imshape)
    SSmom_2 = sp.zeros(imshape)
    SSmom_8 = sp.zeros(imshape)
    SSIpeak = sp.zeros(imshape)
    fiterrormap = sp.zeros(imshape)
    if DGauss:
        im_g2_a = sp.zeros(imshape)
        im_g2_v0 = sp.zeros(imshape)
        im_g2_sigma = sp.zeros(imshape)
        im_g2_a_e = sp.zeros(imshape)
        im_g2_v0_e = sp.zeros(imshape)
        im_g2_sigma_e = sp.zeros(imshape)

    if StoreModel:
        modelcube = sp.zeros(cube.shape)
        if DGauss:
            modelcube_g1 = sp.zeros(cube.shape)
            modelcube_g2 = sp.zeros(cube.shape)
    if (DoBaseline):
        base_a_map = sp.zeros(imshape)
        base_b_map = sp.zeros(imshape)
    dv = abs(velocities[1] - velocities[0])

    
    tasks=range(npix)
    with Pool(n_cores) as pool:
        passpoolresults = list(tqdm(pool.imap(fitter, tasks), total=len(tasks)))
        pool.close()
        pool.join()


    #p = Pool(n_cores)
    #passpoolresults = p.map(fitter, range(npix))



    print( ('Done whole pool'))
    passpoolresults = sp.array(passpoolresults)
#passpoolresults = passpoolresults[passpoolresults!=None]
    for ls in passpoolresults:
        if None in ls:
            continue
        i = int(ls[0])
        j = int(ls[1])


        im_gmom_0[j,i] = ls[2]
        im_g_a[j,i] = ls[3]
        im_g_a_e[j,i] = ls[4]
        im_g_v0[j,i] = ls[5]
        im_g_v0_e[j,i] = ls[6]
        im_g_sigma[j,i] = ls[7]
        im_g_sigma_e[j,i] = ls[8]

        
        SSmom_0[j,i] = ls[9]
        SSmom_1[j,i] = ls[10]
        SSmom_2[j,i] = ls[11]
        SSmom_8[j,i] = ls[12]

        fiterrormap[j,i]=ls[13]
        gaussfit1=ls[14]
        gaussfits=gaussfit1.copy()
        if DGauss:
            gaussfit2=ls[15]
            im_g2_a[j,i]=ls[16]
            im_g2_a_e[j,i]=ls[17]
            im_g2_v0[j,i]=ls[18]
            im_g2_v0_e[j,i]=ls[19]
            im_g2_sigma[j,i]=ls[20]
            im_g2_sigma_e[j,i]=ls[21]
            im_gmom_8[j,i]=ls[22]
            icount=22
            gaussfits += gaussfit2
        else:
            icount=14

        modelspectrum=gaussfits.copy()
        if (DoBaseline):
            base_a_map[j,i]=ls[icount+1]
            base_b_map[j,i]=ls[icount+2]
            baseline=ls[icount+3]
            modelspectrum+=baseline
            icount+=3

        SSIpeak[j,i]=ls[icount+1]
        
        if StoreModel:
            if DGauss:
                if (len(cube.shape) > 3):
                    modelcube[0,:,j,i]=modelspectrum[:]
                    modelcube_g1[0,:,j,i]=gaussfit1[:]
                    modelcube_g2[0,:,j,i]=gaussfit2[:]
                else:
                    modelcube[:,j,i]=modelspectrum[:]
                    modelcube_g1[:,j,i]=gaussfit1[:]
                    modelcube_g2[:,j,i]=gaussfit2[:]
            else:
                if (len(cube.shape) > 3):
                    modelcube[0,:,j,i]=modelspectrum[:]
                else:
                    modelcube[:,j,i]=modelspectrum[:]

    headim = deepcopy(headcube)
    
    if (not 'BMAJ' in headim.keys()):
        print("no beam info, look for extra HDU")
        beamdata = pf.open(cubefile)[1].data
        bmaj=beamdata[0][0]
        bmin=beamdata[0][1]
        bpa=beamdata[0][2]
        headim['BMAJ']=bmaj/3600.
        headim['BMIN']=bmaj/3600.
        headim['BPA']=bmaj


    headim.pop('CTYPE3', None)
    headim.pop('CRVAL3', None)
    headim.pop('CDELT3', None)
    headim.pop('CRPIX3', None)
    headim.pop('CUNIT3', None)
    headim.pop('CUNIT4', None)
    headim.pop('CTYPE4', None)
    headim.pop('CRVAL4', None)
    headim.pop('CDELT4', None)
    headim.pop('CRPIX4', None)
    
    
    head1 = copy(headim)
    head2 = copy(headim)


    head1['BTYPE'] = 'Integrated Intensity'
    head1['BUNIT'] = head1['BUNIT'] + ' km/s'


    head2['BTYPE'] = 'Velocity'
    head2['BUNIT'] = 'km/s'

    import os.path 

    if (not re.search(r"\/$",workdir)):
        workdir+='/'
        print("added trailing back slack to workdir")
        
    if (not os.path.isdir(workdir)):
        os.system("mkdir "+workdir)

    if ShrinkCanvas:
        pf.writeto(workdir+'/'+'datacube.fits',cube,headcube,overwrite=True)
        


    pf.writeto(workdir+'/'+'im_gmom_0.fits',im_gmom_0,head1,overwrite=True)
    
    pf.writeto(workdir+'/'+'im_g_a.fits',im_g_a,headim,overwrite=True)
    pf.writeto(workdir+'/'+'im_g_a_e.fits',im_g_a_e,headim,overwrite=True)
    pf.writeto(workdir+'/'+'im_g_v0.fits',im_g_v0,head2,overwrite=True)
    pf.writeto(workdir+'/'+'im_g_v0_e.fits',im_g_v0_e,head2,overwrite=True)
    pf.writeto(workdir+'/'+'im_g_sigma.fits',im_g_sigma,head2,overwrite=True)
    pf.writeto(workdir+'/'+'im_g_sigma_e.fits',im_g_sigma_e,head2,overwrite=True)
    
    if (DGauss):
        pf.writeto(workdir+'/'+'im_g2_a.fits',im_g2_a,headim,overwrite=True)
        pf.writeto(workdir+'/'+'im_g2_a_e.fits',im_g2_a_e,headim,overwrite=True)
        pf.writeto(workdir+'/'+'im_g2_v0.fits',im_g2_v0,head2,overwrite=True)
        pf.writeto(workdir+'/'+'im_g2_v0_e.fits',im_g2_v0_e,head2,overwrite=True)
        pf.writeto(workdir+'/'+'im_g2_sigma.fits',im_g2_sigma,head2,overwrite=True)
        pf.writeto(workdir+'/'+'im_g2_sigma_e.fits',im_g2_sigma_e,head2,overwrite=True)
        pf.writeto(workdir+'/'+'im_gmom_8.fits',im_gmom_8,headim,overwrite=True)
        if StoreModel:
            pf.writeto(workdir+'/'+'modelcube_g1.fits',modelcube_g1,headcube,overwrite=True)
            pf.writeto(workdir+'/'+'modelcube_g2.fits',modelcube_g2,headcube,overwrite=True)
        
    if (DoBaseline):
        pf.writeto(workdir+'/'+'base_a.fits',base_a_map,head2,overwrite=True)
        pf.writeto(workdir+'/'+'base_b.fits',base_b_map,head2,overwrite=True)

    pf.writeto(workdir+'/'+'Smom_0.fits',SSmom_0,head1,overwrite=True)
    pf.writeto(workdir+'/'+'Smom_1.fits',SSmom_1,head2,overwrite=True)
    pf.writeto(workdir+'/'+'Smom_2.fits',SSmom_2,head2,overwrite=True)
    pf.writeto(workdir+'/'+'Smom_8.fits',SSmom_8,head2,overwrite=True)


    pf.writeto(workdir+'/'+'im_Ipeak.fits',SSIpeak,headim,overwrite=True)

    
    pf.writeto(workdir+'/'+'fiterrormap.fits',fiterrormap,headim,overwrite=True)

    if StoreModel:
        pf.writeto(workdir+'/'+'modelcube.fits',modelcube,headcube,overwrite=True)
    
