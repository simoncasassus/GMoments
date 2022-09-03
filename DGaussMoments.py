import scipy as sp
import astropy.io.fits as pf
from scipy.integrate import simps, quad
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
import re
import scipy.optimize as op
from tqdm import tqdm
import time
import numpy.ma as ma

from operator import itemgetter, attrgetter

#if not sys.warnoptions:
#    import os, warnings
#    #warnings.simplefilter("default") # Change the filter in this process
#    warnings.simplefilter("ignore") # Change the filter in this process
#    #os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses
#    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
#

c_kms = 1.e-3 * const.c.value  ## light speed in km/s


def gaussian(x, a, mu, sigma):
    return a * sp.exp(-.5 * ((x - mu) / sigma)**2.)


def neggaussfit(x, a, mu, sigma, a2, mu2, sigma2):
    model = gaussian(x, a, mu, sigma) + gaussian(x, a2, mu2, sigma2)
    return -model


#def v2(x,a,mu,sigma,a2,mu2,sigma2):
#    print("in v2")
#    return x**2
#

#def chi2_2gauss_wbase_commonsigma(x,a,mu,sigma,a2,mu2,obsspectrum,rmsnoise,baseparams=[0.,0.]):
#    baseline=baseparams[0]*x+baseparams[1]
#    model= baseline+gaussian(x,a,mu,sigma) + gaussian(x,a2,mu2,sigma)
#    aux = (np.abs(obsspectrum-model))**2.
#    chi2 = sp.sum(aux)
#    chi2 = chi2/rmsnoise**2.
#    return chi2
#


def dgauss_wbase(x, a1, mu1, sigma1, a2, mu2, sigma2, base_a, base_b):
    baseline = base_a * x + base_b
    model = baseline + gaussian(x, a1, mu1, sigma1) + gaussian(
        x, a2, mu2, sigma2)
    return model


def sgauss_wbase(x, a1, mu1, sigma1, base_a, base_b):
    baseline = base_a * x + base_b
    model = baseline + gaussian(x, a1, mu1, sigma1)
    return model


#def dgauss_wbase_commonsigma(x,a,mu,sigma,a2,mu2,base_a,base_b):
#    baseline=base_a*x+base_b
#    model= baseline+gaussian(x,a,mu,sigma) + gaussian(x,a2,mu2,sigma)
#    return model
#
#def chi2_2gauss_wbase(x,a,mu,sigma,a2,mu2,sigma2,obsspectrum,rmsnoise,baseparams=[0.,0.]):
#    baseline=baseparams[0]*x+baseparams[1]
#    model= baseline+gaussian(x,a,mu,sigma) + gaussian(x,a2,mu2,sigma2)
#    aux = (np.abs(obsspectrum-model))**2.
#    chi2 = sp.sum(aux)
#    chi2 = chi2/rmsnoise**2.
#    return chi2
#
#
#def chi2_2gauss(x,a,mu,sigma,a2,mu2,sigma2,obsspectrum,rmsnoise):
#    model= gaussian(x,a,mu,sigma) + gaussian(x,a2,mu2,sigma2)
#    aux = (np.abs(obsspectrum-model))**2.
#    chi2 = sp.sum(aux)
#    chi2 = chi2/rmsnoise**2.
#    return chi2
#
#def chi2_gauss_wbase(x,a,mu,sigma,obsspectrum,rmsnoise,baseparams=[0.,0.]):
#    baseline=baseparams[0]*x+baseparams[1]
#    gauss = baseline+gaussian(x,a,mu,sigma)
#    aux = (np.abs(obsspectrum-gauss))**2.
#    chi2 = sp.sum(aux)
#    chi2 = chi2/rmsnoise**2.
#    return chi2
#
#def chi2_gauss(x,a,mu,sigma,obsspectrum,rmsnoise):
#    gauss = gaussian(x,a,mu,sigma)
#    aux = (np.abs(obsspectrum-gauss))**2.
#    chi2 = sp.sum(aux)
#    chi2 = chi2/rmsnoise**2.
#    return chi2
#


def fitter(alos):
    #if ( ((float(npix)/float(n))-int(float(npix)/float(n))) == 0):
    #    print( (float(npix)/float(n)),"%")

    #i = int(n/side_pix)
    #j = int(n%side_pix)
    #i += x_i
    #j += y_i

    j = alos[0]
    i = alos[1]

    #example near center with signal
    #i=1105
    #j=1061

    if len(cube.shape) > 3:
        lineofsight_spectrum = cube[0, :, j, i]
    else:
        lineofsight_spectrum = cube[:, j, i]

    #mask=lineofsight_spectrum > -0.01 * lineofsight_spectrum.max() ##mask for values too negative.

    if (isinstance(MaskCube, np.ndarray)):
        masklos = np.logical_not(ma.make_mask(MaskCube[:, j, i]))
        lineofsight_spectrum[masklos] = 0.

    if (lineofsight_spectrum.max() <= 0):
        return [None]

    mask = np.ones((len(lineofsight_spectrum), ), dtype=bool)

    if BadChannels:
        ibadchan1 = BadChannels[0]
        ibadchan2 = BadChannels[1]
        mask[ibadchan1:ibadchan2] = False

    selected_velocities = velocities[mask]
    selected_losspectrum = lineofsight_spectrum[mask]
    Amp_init = selected_losspectrum.max()  #- selected_losspectrum.mean()
    if ((Amp_init <= 0) or (np.isnan(Amp_init))):
        return [None]

    #v0_init = selected_velocities[selected_losspectrum==selected_losspectrum.max()]
    v0_init = selected_velocities[np.argmax(selected_losspectrum)]
    sigma_init = 3. * dv
    sigma_init = dv
    sigma_max = 10.
    if LocalNoise:
        ## take the error as the rmsnoise far from the line
        localnoise = selected_losspectrum[(selected_velocities < v0_init - 1.)
                                          |
                                          (selected_velocities > v0_init + 1.)]
        localrmsnoise = np.std(localnoise)

    if (DoClip):
        lineofsight_spectrum[np.where(lineofsight_spectrum < 0)] = 0.
        selected_losspectrum[np.where(selected_losspectrum < 0)] = 0.

    a1_init = Amp_init
    mu1_init = v0_init
    sigma1_init = sigma_init
    maxslope = Amp_init / np.fabs(
        (np.max(selected_velocities) - np.min(selected_velocities)))
    base_a_init = 0.
    base_b_init = 0.

    if (dv < 0.):
        sys.exit("something is wrong with dv")

    limit_a1 = [0., 5. * Amp_init]  #Bounds for pars
    limit_mu1 = [velocities.min() - 1.0, velocities.max() + 1.0]
    limit_sigma1 = [dv / 3., sigma_max]
    limit_base_a = [-maxslope, maxslope]
    limit_base_b = [0., selected_losspectrum.max()]

    fallback_error_a1 = 1E20  # limit_a1[1]
    fallback_error_mu1 = 1E20
    fallback_error_sigma1 = 1E20
    fallback_error_base_a = maxslope
    fallback_error_base_b = 1E20  # rmsnoise

    if DoBaseline:
        func = lambda x, a1, mu1, sigma1, base_a, base_b: sgauss_wbase(
            x, a1, mu1, sigma1, base_a, base_b)
        p0 = np.array(
            [a1_init, mu1_init, sigma1_init, base_a_init, base_b_init])
        bounds = (limit_a1, limit_mu1, limit_sigma1, limit_base_a,
                  limit_base_b)
        fallback_errors = np.array(
            (fallback_error_a1, fallback_error_mu1, fallback_error_sigma1,
             fallback_error_base_a, fallback_error_base_b))
    else:
        func = lambda x, a1, mu1, sigma1: sgauss_wbase(
            x, a1, mu1, sigma1, base_a_init, base_b_init)
        p0 = np.array([a1_init, mu1_init, sigma1_init])
        bounds = (limit_a1, limit_mu1, limit_sigma1)
        fallback_errors = np.array(
            (fallback_error_a1, fallback_error_mu1, fallback_error_sigma1))

    xdata = selected_velocities
    ydata = selected_losspectrum

    lowlimits = []
    uplimits = []
    for abounds in bounds:
        lowlimits.append(abounds[0])
        uplimits.append(abounds[1])

    try:
        if LocalNoise:
            popt, pcov = op.curve_fit(func,
                                      xdata,
                                      ydata,
                                      sigma=np.ones(len(ydata)) *
                                      localrmsnoise,
                                      p0=p0,
                                      bounds=[lowlimits, uplimits],
                                      absolute_sigma=True)
        else:
            #popt, pcov = op.curve_fit(func, xdata, ydata,sigma=np.ones(len(ydata))*rmsnoise,p0=p0,bounds=[lowlimits,uplimits],absolute_sigma=False)
            if (rmsnoise < 0.):
                popt, pcov = op.curve_fit(func,
                                          xdata,
                                          ydata,
                                          p0=p0,
                                          bounds=[lowlimits, uplimits],
                                          absolute_sigma=False)
            else:
                popt, pcov = op.curve_fit(func,
                                          xdata,
                                          ydata,
                                          sigma=np.ones(len(ydata)) * rmsnoise,
                                          p0=p0,
                                          bounds=[lowlimits, uplimits],
                                          absolute_sigma=True)

                #popt, pcov = op.curve_fit(func, xdata, ydata,sigma=np.ones(len(ydata))*rmsnoise,p0=p0)

        if (np.any(np.isnan(popt)) or np.any(np.diag(pcov) < 0.)):
            # print("invalid values in variances",np.diag(pcov))
            popt = p0.copy()
            perr = fallback_errors
        else:
            perr = np.sqrt(np.diag(pcov))
    except:
        #print("Error - curve_fit failed")
        popt = p0.copy()
        perr = fallback_errors

    optimresult1 = {}

    #optimresult['a']={value:popt[0],error:perr[0]}
    optimresult1['values'] = {}
    optimresult1['errors'] = {}
    #paramnames=['a1','mu1','sigma1','a2','mu2','sigma2','base_a','base_b']
    setofparamnames1 = ['a1', 'mu1',
                        'sigma1']  #,'a2','mu2','sigma2','base_a','base_b']
    for param in enumerate(setofparamnames1):
        iparam = param[0]
        aparam = param[1]
        optimresult1['values'][aparam] = popt[iparam]
        optimresult1['errors'][aparam] = perr[iparam]

    g_amp = optimresult1['values']['a1']
    g_v0 = optimresult1['values']['mu1']
    g_sigma = optimresult1['values']['sigma1']

    g_amp_e = optimresult1['errors']['a1']
    g_v0_e = optimresult1['errors']['mu1']
    g_sigma_e = optimresult1['errors']['sigma1']

    if (DGauss):
        # velorange=velocities.min()-1.0, velocities.max()+1.0

        #a1_init=Amp_init
        #mu1_init=v0_init
        #sigma1_init=sigma_init

        a1_init = g_amp
        mu1_init = g_v0
        sigma1_init = g_sigma

        #v0_init2 = v0_init
        #a2_init=Amp_init*0.5

        mu2_init = g_v0
        a2_init = g_amp * 0.5
        sigma2_init = g_sigma * 0.8

        if (Randomize):
            #v0_init2 = v0_init + 3.*dv*(np.random.random()-0.5) # selected_velocities[selected_losspectrum==selected_losspectrum.max()]
            #a2_init=Amp_init*0.2 #0.5 # np.random.random()

            sigma1_init = g_sigma * (0.8 + 0.2 *
                                     (np.random.random() - 0.5) / 0.5)

            v0_init2 = g_v0 + 3. * dv * (np.random.random() - 0.5)  #
            if ((v0_init2 < limit_mu1[0]) or (v0_init2 > limit_mu1[1])):
                v0_init2 = v0_init
            mu2_init = v0_init2
            a2_init = g_amp * (0.4 + 0.2 * (np.random.random() - 0.5) / 0.5)
            sigma2_init = g_sigma * (0.6 + 0.2 *
                                     (np.random.random() - 0.5) / 0.5)

        limit_a2 = [0., Amp_init]  #Bounds for pars
        limit_mu2 = [velocities.min() - 1.0, velocities.max() + 1.0]
        limit_sigma2 = [dv / 3., sigma_max]

        fallback_error_a2 = fallback_error_a1
        fallback_error_mu2 = fallback_error_mu1
        fallback_error_sigma2 = fallback_error_sigma1

        if (CommonSigma):
            if DoBaseline:
                func = lambda x, a1, mu1, sigma1, a2, mu2, base_a, base_b: dgauss_wbase(
                    x, a1, mu1, sigma1, a2, mu2, sigma1, base_a, base_b)
                p0 = np.array([
                    a1_init, mu1_init, sigma1_init, a2_init, mu2_init,
                    base_a_init, base_b_init
                ])
                bounds = (limit_a1, limit_mu1, limit_sigma1, limit_a2,
                          limit_mu2, limit_base_a, limit_base_b)
                fallback_errors = np.array(
                    (fallback_error_a1, fallback_error_mu1,
                     fallback_error_sigma1, fallback_error_a2,
                     fallback_error_mu2, fallback_error_base_a,
                     fallback_error_base_b))
            else:
                func = lambda x, a1, mu1, sigma1, a2, mu2: dgauss_wbase(
                    x, a1, mu1, sigma1, a2, mu2, sigma1, base_a_init,
                    base_b_init)
                p0 = np.array(
                    [a1_init, mu1_init, sigma1_init, a2_init, mu2_init])
                bounds = (limit_a1, limit_mu1, limit_sigma1, limit_a2,
                          limit_mu2)
                fallback_errors = np.array(
                    (fallback_error_a1, fallback_error_mu1,
                     fallback_error_sigma1, fallback_error_a2,
                     fallback_error_mu2))
        else:
            if DoBaseline:
                func = lambda x, a1, mu1, sigma1, a2, mu2, sigma2, base_a, base_b: dgauss_wbase(
                    x, a1, mu1, sigma1, a2, mu2, sigma2, base_a, base_b)
                p0 = np.array([
                    a1_init, mu1_init, sigma1_init, a2_init, mu2_init,
                    sigma2_init, base_a_init, base_b_init
                ])
                bounds = (limit_a1, limit_mu1, limit_sigma1, limit_a2,
                          limit_mu2, limit_sigma2, limit_base_a, limit_base_b)
                fallback_errors = np.array(
                    (fallback_error_a1, fallback_error_mu1,
                     fallback_error_sigma1, fallback_error_a2,
                     fallback_error_mu2, fallback_error_sigma2,
                     fallback_error_base_a, fallback_error_base_b))
            else:
                func = lambda x, a1, mu1, sigma1, a2, mu2, sigma2: dgauss_wbase(
                    x, a1, mu1, sigma1, a2, mu2, sigma2, base_a_init,
                    base_b_init)
                p0 = np.array([
                    a1_init, mu1_init, sigma1_init, a2_init, mu2_init,
                    sigma2_init
                ])
                bounds = (limit_a1, limit_mu1, limit_sigma1, limit_a2,
                          limit_mu2, limit_sigma2)
                fallback_errors = np.array(
                    (fallback_error_a1, fallback_error_mu1,
                     fallback_error_sigma1, fallback_error_a2,
                     fallback_error_mu2, fallback_error_sigma2))

    xdata = selected_velocities
    ydata = selected_losspectrum

    lowlimits = []
    uplimits = []
    for abounds in bounds:
        lowlimits.append(abounds[0])
        uplimits.append(abounds[1])

    try:
        if LocalNoise:
            popt, pcov = op.curve_fit(func,
                                      xdata,
                                      ydata,
                                      sigma=np.ones(len(ydata)) *
                                      localrmsnoise,
                                      p0=p0,
                                      bounds=[lowlimits, uplimits],
                                      absolute_sigma=True)
        else:
            #popt, pcov = op.curve_fit(func, xdata, ydata,sigma=np.ones(len(ydata))*rmsnoise,p0=p0,bounds=[lowlimits,uplimits],absolute_sigma=False)
            if (rmsnoise < 0.):
                popt, pcov = op.curve_fit(func,
                                          xdata,
                                          ydata,
                                          p0=p0,
                                          bounds=[lowlimits, uplimits],
                                          absolute_sigma=False)
            else:
                popt, pcov = op.curve_fit(func,
                                          xdata,
                                          ydata,
                                          sigma=np.ones(len(ydata)) * rmsnoise,
                                          p0=p0,
                                          bounds=[lowlimits, uplimits],
                                          absolute_sigma=True)

                #popt, pcov = op.curve_fit(func, xdata, ydata,sigma=np.ones(len(ydata))*rmsnoise,p0=p0)

        if (np.any(np.isnan(popt)) or np.any(np.diag(pcov) < 0.)):
            # print("invalid values in variances",np.diag(pcov))
            popt = p0.copy()
            perr = fallback_errors
        else:
            perr = np.sqrt(np.diag(pcov))
    except:
        #print("Error - curve_fit failed")
        popt = p0.copy()
        perr = fallback_errors

    optimresult = {}

    #optimresult['a']={value:popt[0],error:perr[0]}
    optimresult['values'] = {}
    optimresult['errors'] = {}
    #paramnames=['a1','mu1','sigma1','a2','mu2','sigma2','base_a','base_b']
    setofparamnames = ['a1', 'mu1',
                       'sigma1']  #,'a2','mu2','sigma2','base_a','base_b']
    for param in enumerate(setofparamnames):
        iparam = param[0]
        aparam = param[1]
        optimresult['values'][aparam] = popt[iparam]
        optimresult['errors'][aparam] = perr[iparam]

    g_amp = optimresult['values']['a1']
    g_v0 = optimresult['values']['mu1']
    g_sigma = optimresult['values']['sigma1']

    # KEEEP SINGLE GAUSSIAN ERRORS BECAUSE OF DEGENERACIES THAT SHOOT UP THE HESSIAN ERRORS
    g_amp_e = optimresult1['errors']['a1']
    g_v0_e = optimresult1['errors']['mu1']
    g_sigma_e = optimresult1['errors']['sigma1']

    if (ViewSingleSpectrum):
        if (LocalNoise):
            print("using LocalNoise", localrmsnoise)
        elif (rmsnoise > 0.):
            print("using fixed rms noised ", rmsnoise)
        else:
            print("using default red. chi2 = 1 noise normalization")

        print("g_amp", g_amp, "+-", g_amp_e)
        print("g_v0", g_v0, "+-", g_v0_e)
        print("g_sigma", g_sigma, "+-", g_sigma_e)

    if (g_sigma < 0):
        # print("g_sigma negative: ",g_sigma)
        g_sigma = sigma1_init
        g_sigma_e = fallback_error_sigma1

    gaussfit1 = gaussian(velocities, g_amp, g_v0, g_sigma)
    fit1 = gaussfit1

    if (DGauss):
        pars = popt.copy()
        err_pars = perr.copy()
        if CommonSigma:
            #pars = [optimresult['values']['a'], optimresult['values']['mu'], optimresult['values']['sigma'],optimresult['values']['a2'], optimresult['values']['mu2'], optimresult['values']['sigma']]   # pars for best fit
            #err_pars = [optimresult['errors']['a'], optimresult['errors']['mu'], optimresult['errors']['sigma'],optimresult['errors']['a2'], optimresult['errors']['mu2'], optimresult['errors']['sigma']]  #error in pars
            pars = np.append(pars, pars[2])
            err_pars = np.append(err_pars, pars[2])

        amps = [[pars[0], 0], [pars[3], 3]]
        amps_sorted = sorted(amps, key=itemgetter(0))
        i_G1 = amps_sorted[-1][1]
        i_G2 = amps_sorted[0][1]

        g_amp = pars[i_G1]
        g_v0 = pars[i_G1 + 1]
        g_sigma = pars[i_G1 + 2]

        # KEEEP SINGLE GAUSSIAN ERRORS BECAUSE OF DEGENERACIES THAT SHOOT UP THE HESSIAN ERRORS
        g1_amp_e = err_pars[i_G1]
        g1_v0_e = err_pars[i_G1 + 1]
        g1_sigma_e = err_pars[i_G1 + 2]

        gaussfit1 = gaussian(velocities, g_amp, g_v0, g_sigma)

        g2_amp = pars[i_G2]
        g2_v0 = pars[i_G2 + 1]
        g2_sigma = pars[i_G2 + 2]
        g2_amp_e = err_pars[i_G2]
        g2_v0_e = err_pars[i_G2 + 1]
        g2_sigma_e = err_pars[i_G2 + 2]

        if (ViewSingleSpectrum):
            print("g2_amp", g2_amp, "+-", g2_amp_e)
            print("g2_v0", g2_v0, "+-", g2_v0_e)
            print("g2_sigma", g2_sigma, "+-", g2_sigma_e)

        gaussfit2 = gaussian(velocities, g2_amp, g2_v0, g2_sigma)

        fit1 = gaussfit1 + gaussfit2

        vpeak = velocities[np.argmax(fit1)]
        ComputeG8 = True
        if ComputeG8:
            vpeak_init = vpeak

            f_vpeak = lambda vmax: neggaussfit(vmax, g_amp, g_v0, g_sigma,
                                               g2_amp, g2_v0, g2_sigma)

            #mvpeak = Minuit(f_vpeak, vmax=vpeak_init,  #initial guess
            #                errordef=1,                      # error
            #                error_vmax=dv*0.01,
            #                #limit_vmax=(velocities.min()-1.0, velocities.max()+1.0),
            #                limit_vmax=(np.min(selected_velocities), np.max(selected_velocities)), #Bounds for pars
            #                print_level=0,
            #                )
            #mvpeak.migrad()
            #
            #vpeakMinuit=mvpeak.values['vmax']

            res = op.minimize(f_vpeak, vpeak_init)
            vpeak = res.x
            #print("Scipy optimize",vpeak," Minuit optimize",vpeakMinuit)

    # gmom_0 = abs(simps(fit1,velocities))
    gmom_0 = np.sqrt(
        2. * np.pi) * g_sigma * g_amp  # abs(simps(fit1,velocities))

    popt_nobase = deepcopy(popt)
    if (DoBaseline):
        popt_nobase[-1] = 0.
        popt_nobase[-2] = 0.

    popt_nobase_args = popt_nobase.tolist()
    fit2 = func(velocities, *popt_nobase_args)

    gmom_0_simps = abs(simps(fit2, velocities))
    #args=popt_nobase_args

    if DGauss:
        v1 = min(velocities[0], velocities[-1])
        v2 = max(velocities[0], velocities[-1])

        simpssign = 1.
        if (velocities[1] < velocities[0]):
            simpssign = -1.

        func_wargs = lambda v: func(v, *popt_nobase_args)
        if DoQuad:
            integ_result = quad(func_wargs, v1, v2, full_output=1)
            gmom_0_quad = integ_result[0]
        else:
            evalfunc = func_wargs(velocities)
            gmom_0_quad = simpssign * simps(evalfunc, velocities)

        first_mom_func_wargs = lambda v: v * func(v, *popt_nobase_args)
        if DoQuad:
            integ_result = quad(first_mom_func_wargs, v1, v2, full_output=1)
            Num_mom1 = integ_result[0]
        else:
            evalfunc = first_mom_func_wargs(velocities)
            Num_mom1 = simpssign * simps(evalfunc, velocities)

        if (gmom_0_quad <= 1E-20):
            gmom_1_quad = fallback_error_mu1
        else:
            gmom_1_quad = Num_mom1 / gmom_0_quad

        second_mom_func_wargs = lambda v: (
            (v - gmom_1_quad)**2 * func(v, *popt_nobase_args))

        if DoQuad:
            integ_result = quad(second_mom_func_wargs, v1, v2, full_output=1)
            Num_mom2 = integ_result[0]
        else:
            evalfunc = second_mom_func_wargs(velocities)
            Num_mom2 = simpssign * simps(evalfunc, velocities)

        if ((gmom_0_quad <= 0) or (Num_mom2 <= 0)):
            gmom_2_quad = 0.
        else:
            gmom_2_quad = np.sqrt((Num_mom2) / gmom_0_quad)

    #gmom_0_quad = np.sqrt( 2. * np.pi) * g_sigma * g_amp

    #print("gmom_0s ",gmom_0,gmom_0_b,gmom_0_c)

    if (DoBaseline):
        base_a = popt[-2]
        base_b = popt[-1]
        baseline = base_a * velocities + base_b
        fit1 += baseline

    if (ViewSingleSpectrum):
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots()

        save_prof = np.zeros((len(velocities), 2))
        save_prof[:, 0] = velocities
        save_prof[:, 1] = lineofsight_spectrum
        fileout_spectrum = 'LOSspectrum_' + str(LOS['dalpha']) + '_' + str(
            LOS['ddelta']) + '.dat'
        np.savetxt(fileout_spectrum, save_prof)

        theplot = ax1.plot(velocities, fit1, label='model', lw=4., alpha=0.5)
        ax1.plot(velocities, lineofsight_spectrum, label='obs')
        plt.legend()
        plt.show()
        return

    fiterror = np.std(fit1 - lineofsight_spectrum)

    #fitinit = gaussian(velocities, Amp_init,v0_init,sigma_init)
    #plt.plot(velocities, signal)
    #plt.plot(velocities, fit1)
    #plt.plot(velocities, fitinit)
    #plt.show()

    # print( "vel range:",np.min(velocities),np.max(velocities),"best fit",pars[1])
    ic = np.argmin(abs(velocities - g_v0))
    if (g_sigma < sigma_init):
        Delta_i = sigma_init / dv
    else:
        Delta_i = g_sigma / dv

    nsigrange = 5

    i0 = int(ic - nsigrange * Delta_i)
    if (i0 < 0):
        i0 = 0
    i1 = int(ic + nsigrange * Delta_i)
    if (i1 > (len(velocities) - 1)):
        i1 = (len(velocities) - 1)
    j0 = int(ic - nsigrange * Delta_i)
    if (j0 < 0):
        j0 = 0
    j1 = int(ic + nsigrange * Delta_i)
    if (j1 > (len(velocities) - 1)):
        j1 = (len(velocities) - 1)

    #print( "i0",i0,"i1",i1,"j0",j0,"j1",j1)

    sign = 1.
    if (velocities[1] < velocities[0]):
        sign = -1
    Smom_0 = sign * simps(lineofsight_spectrum[i0:i1], velocities[i0:i1])
    Smom_1 = sign * simps(lineofsight_spectrum[i0:i1] * velocities[i0:i1],
                          velocities[i0:i1])
    subvelo = velocities[i0:i1]
    Smom_8 = subvelo[np.argmax(lineofsight_spectrum[i0:i1])]
    Smax = np.max(lineofsight_spectrum[i0:i1])

    if (abs(Smom_0) > 0):
        Smom_1 /= Smom_0
        if (Smom_0 > 0):
            var = sign * simps(
                lineofsight_spectrum[i0:i1] *
                (velocities[i0:i1] - Smom_1)**2, velocities[i0:i1])
            if (var > 0):
                Smom_2 = np.sqrt(var / Smom_0)
            else:
                Smom_2 = -1E6
        else:
            Smom_2 = -1E6
    else:
        Smom_1 = -1E6
        Smom_2 = -1E6

    passresults = [
        i, j, gmom_0, g_amp, g_amp_e, g_v0, g_v0_e, g_sigma, g_sigma_e, Smom_0,
        Smom_1, Smom_2, Smom_8, fiterror, gaussfit1
    ]
    if DGauss:
        passresults.extend([
            gaussfit2, g2_amp, g2_amp_e, g2_v0, g2_v0_e, g2_sigma, g2_sigma_e,
            vpeak, gmom_0_quad, gmom_1_quad, gmom_2_quad
        ])
    if (DoBaseline):
        passresults.extend([base_a, base_b, baseline])
    passresults.extend([
        Smax,
    ])

    return passresults


def exec_Gfit(cubefile,
              workdir=None,
              wBaseline=False,
              n_cores=30,
              zoom_area=-1.,
              Noise=-1.0,
              Clip=False,
              DoubleGauss=False,
              StoreModel=False,
              Randomize2ndGauss=True,
              ShrinkCanvas=True,
              UseCommonSigma=False,
              PassRestFreq=-1,
              UseLOSNoise=False,
              singleLOS=False,
              MaskChannels=False,
              cubemask=False,
              PerformAccurateInteg=True):
    #Region=True: zoom into central region, defined as nx/2., with half side zoom_area
    #zoom_area=1.2 # arcsec

    # PerformAccurateInteg=False, speed up by 10%

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
    global LocalNoise
    global ViewSingleSpectrum
    global LOS
    global BadChannels
    global DoQuad
    global MaskCube

    start_time = time.time()
    print("start Curve_fit:", time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.gmtime()))

    DoClip = Clip
    rmsnoise = Noise
    DoBaseline = wBaseline
    DGauss = DoubleGauss
    Randomize = Randomize2ndGauss
    CommonSigma = UseCommonSigma
    LocalNoise = UseLOSNoise
    ViewSingleSpectrum = False
    if (singleLOS):
        ViewSingleSpectrum = True
        LOS = singleLOS
    BadChannels = MaskChannels
    DoQuad = PerformAccurateInteg

    print("Collapsing cube from file:", cubefile)
    print(">>>>", cubefile)

    datacube = pf.open(cubefile)[0].data
    datahdr = pf.open(cubefile)[0].header

    print("datacube.shape", datacube.shape)

    MaskCube = False
    if cubemask:
        MaskCube = pf.open(cubemask)[0].data
        print("MaskCube.shape", MaskCube.shape)
        #print(type(MaskCube))

        #datahdr = pf.open(cubemask)[0].header

    # if len(cube.shape)>3:
    #    # cube = cube[1:]
    #    cube = cube[0,:,:,:]
    # print("cube.shape",cube.shape)

    # cube = sp.swapaxes(cube,0,2)
    # cube = sp.swapaxes(cube,0,1)

    #dnu = datahdr['CDELT3']
    #len_nu = datahdr['NAXIS3']
    #nui = datahdr['CRVAL3']- (datahdr['CRPIX3']-1)*dnu
    #nuf = nui + (len_nu-1)*dnu
    #nu = sp.linspace(nui, nuf, len_nu)

    nu = (np.arange(datahdr['NAXIS3']) -
          (datahdr['CRPIX3'] - 1)) * datahdr['CDELT3'] + datahdr['CRVAL3']

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

    icenter = int(datahdr['CRPIX1'] - 1.)
    if (zoom_area > 0.):
        halfside_pix = int(zoom_area / (3600. * datahdr['CDELT2']))
        x_i = icenter - halfside_pix
        y_i = icenter - halfside_pix
        x_f = icenter + halfside_pix + 1
        y_f = icenter + halfside_pix + 1
        side_pix = 2 * halfside_pix + 1  # NO NEED TO Resamp WITH ODD NUMBER OF  PIXELS
    else:
        x_i = 0
        y_i = 0
        x_f = datahdr['NAXIS1'] - 1 + 1
        y_f = datahdr['NAXIS1'] - 1 + 1
        side_pix = datahdr['NAXIS1']

    print("x_i", x_i, "x_f", x_f)

    npix = int(side_pix**2)
    print("npix", npix)

    headcube = deepcopy(datahdr)
    if ShrinkCanvas:
        if (len(datacube.shape) > 3):
            cube = datacube[0, :, y_i:y_f, x_i:x_f]
            headcube.pop('CUNIT4', None)
            headcube.pop('CTYPE4', None)
            headcube.pop('CRVAL4', None)
            headcube.pop('CDELT4', None)
            headcube.pop('CRPIX4', None)
        else:
            cube = datacube[:, y_i:y_f, x_i:x_f]

        imshape = cube.shape[1:]
        headcube['CRPIX1'] = headcube['CRPIX1'] - x_i
        headcube['CRPIX2'] = headcube['CRPIX2'] - y_i

        x_i = 0
        y_i = 0
        x_f = side_pix
    else:
        cube = datacube
        if (len(datacube.shape) > 3):
            imshape = cube.shape[2:]
        else:
            imshape = cube.shape[1:]

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
        im_gmom_0 = sp.zeros(imshape)
        im_gmom_1 = sp.zeros(imshape)
        im_gmom_2 = sp.zeros(imshape)

    if StoreModel:
        modelcube = sp.zeros(cube.shape)
        if DGauss:
            modelcube_g1 = sp.zeros(cube.shape)
            modelcube_g2 = sp.zeros(cube.shape)
    if (DoBaseline):
        base_a_map = sp.zeros(imshape)
        base_b_map = sp.zeros(imshape)
    dv = abs(velocities[1] - velocities[0])

    if (singleLOS):
        dalpha = singleLOS['dalpha']
        ddelta = singleLOS['ddelta']
        i = int(headcube['CRPIX1'] + dalpha / (3600. * headcube['CDELT1']))
        j = int(headcube['CRPIX1'] + ddelta / (3600. * headcube['CDELT2']))
        print("SINGLE LOS AT PIXEL: i ", i, " j ", j)
        fitter([j, i])
        return

    tasks = list(range(npix))
    for n in tasks:
        i = int(n / side_pix)
        j = int(n % side_pix)
        i += x_i
        j += y_i
        tasks[n] = [j, i]

    print("n_cores:", n_cores)
    #with Pool(processes=n_cores) as pool:
    with Pool(processes=n_cores) as pool:
        # with Pool(n_cores) as pool:
        passpoolresults = list(tqdm(pool.imap(fitter, tasks),
                                    total=len(tasks)))
        pool.close()
        pool.join()

    #p = Pool(n_cores)
    #passpoolresults = p.map(fitter, range(npix))

    print(('Done whole pool'))
    passpoolresults = sp.array(passpoolresults)

    for alospass in passpoolresults:
        if alospass[0] is None:
            continue

        i = int(alospass[0])
        j = int(alospass[1])

        im_gmom_0[j, i] = alospass[2]
        im_g_a[j, i] = alospass[3]
        im_g_a_e[j, i] = alospass[4]
        im_g_v0[j, i] = alospass[5]
        im_g_v0_e[j, i] = alospass[6]
        im_g_sigma[j, i] = alospass[7]
        im_g_sigma_e[j, i] = alospass[8]

        SSmom_0[j, i] = alospass[9]
        SSmom_1[j, i] = alospass[10]
        SSmom_2[j, i] = alospass[11]
        SSmom_8[j, i] = alospass[12]

        fiterrormap[j, i] = alospass[13]
        gaussfit1 = alospass[14]
        gaussfits = gaussfit1.copy()
        if DGauss:
            gaussfit2 = alospass[15]
            im_g2_a[j, i] = alospass[16]
            im_g2_a_e[j, i] = alospass[17]
            im_g2_v0[j, i] = alospass[18]
            im_g2_v0_e[j, i] = alospass[19]
            im_g2_sigma[j, i] = alospass[20]
            im_g2_sigma_e[j, i] = alospass[21]
            im_gmom_8[j, i] = alospass[22]
            im_gmom_0[j, i] = alospass[23]
            im_gmom_1[j, i] = alospass[24]
            im_gmom_2[j, i] = alospass[25]
            icount = 25
            gaussfits += gaussfit2
        else:
            icount = 14

        modelspectrum = gaussfits.copy()
        if (DoBaseline):
            base_a_map[j, i] = alospass[icount + 1]
            base_b_map[j, i] = alospass[icount + 2]
            baseline = alospass[icount + 3]
            modelspectrum += baseline
            icount += 3

        SSIpeak[j, i] = alospass[icount + 1]

        if StoreModel:
            if DGauss:
                if (len(cube.shape) > 3):
                    modelcube[0, :, j, i] = modelspectrum[:]
                    modelcube_g1[0, :, j, i] = gaussfit1[:]
                    modelcube_g2[0, :, j, i] = gaussfit2[:]
                else:
                    modelcube[:, j, i] = modelspectrum[:]
                    modelcube_g1[:, j, i] = gaussfit1[:]
                    modelcube_g2[:, j, i] = gaussfit2[:]
            else:
                if (len(cube.shape) > 3):
                    modelcube[0, :, j, i] = modelspectrum[:]
                else:
                    modelcube[:, j, i] = modelspectrum[:]

    headim = deepcopy(headcube)

    if (not 'BMAJ' in headim.keys()):
        print("no beam info, look for extra HDU")
        hdulist = pf.open(cubefile)
        if (len(hdulist) > 1):
            beamdata = hdulist[1].data
            bmaj = beamdata[0][0]
            bmin = beamdata[0][1]
            bpa = beamdata[0][2]
            headim['BMAJ'] = bmaj / 3600.
            headim['BMIN'] = bmin / 3600.
            headim['BPA'] = bmaj

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

    if (workdir is None):
        cubefilebasename = os.path.basename(cubefile)
        workdir = re.sub('.fits', '/', cubefilebasename)
        if DGauss:
            prefix = 'dgaussmoments_'
        else:
            prefix = 'sgaussmoments_'
        if LocalNoise:
            prefix += 'LOSNoise_'

        workdir = prefix + workdir
        os.system("rm -rf " + workdir)

    if (not re.search(r"\/$", workdir)):
        workdir += '/'
        print("added trailing back slack to workdir")

    if (not os.path.isdir(workdir)):
        os.system("mkdir " + workdir)
        print("opened workdir: ", workdir)

    if ShrinkCanvas:
        pf.writeto(workdir + '/' + 'datacube.fits',
                   cube,
                   headcube,
                   overwrite=True)

    pf.writeto(workdir + '/' + 'im_gmom_0.fits',
               im_gmom_0,
               head1,
               overwrite=True)

    pf.writeto(workdir + '/' + 'im_g_a.fits', im_g_a, headim, overwrite=True)
    pf.writeto(workdir + '/' + 'im_g_a_e.fits',
               im_g_a_e,
               headim,
               overwrite=True)
    pf.writeto(workdir + '/' + 'im_g_v0.fits', im_g_v0, head2, overwrite=True)
    pf.writeto(workdir + '/' + 'im_g_v0_e.fits',
               im_g_v0_e,
               head2,
               overwrite=True)
    pf.writeto(workdir + '/' + 'im_g_sigma.fits',
               im_g_sigma,
               head2,
               overwrite=True)
    pf.writeto(workdir + '/' + 'im_g_sigma_e.fits',
               im_g_sigma_e,
               head2,
               overwrite=True)

    if (DGauss):
        pf.writeto(workdir + '/' + 'im_g2_a.fits',
                   im_g2_a,
                   headim,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'im_g2_a_e.fits',
                   im_g2_a_e,
                   headim,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'im_g2_v0.fits',
                   im_g2_v0,
                   head2,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'im_g2_v0_e.fits',
                   im_g2_v0_e,
                   head2,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'im_g2_sigma.fits',
                   im_g2_sigma,
                   head2,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'im_g2_sigma_e.fits',
                   im_g2_sigma_e,
                   head2,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'im_gmom_8.fits',
                   im_gmom_8,
                   headim,
                   overwrite=True)

        pf.writeto(workdir + '/' + 'im_gmom_0.fits',
                   im_gmom_0,
                   head1,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'im_gmom_1.fits',
                   im_gmom_1,
                   head2,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'im_gmom_2.fits',
                   im_gmom_2,
                   head2,
                   overwrite=True)

        if StoreModel:
            pf.writeto(workdir + '/' + 'modelcube_g1.fits',
                       modelcube_g1,
                       headcube,
                       overwrite=True)
            pf.writeto(workdir + '/' + 'modelcube_g2.fits',
                       modelcube_g2,
                       headcube,
                       overwrite=True)

    if (DoBaseline):
        pf.writeto(workdir + '/' + 'base_a.fits',
                   base_a_map,
                   head2,
                   overwrite=True)
        pf.writeto(workdir + '/' + 'base_b.fits',
                   base_b_map,
                   head2,
                   overwrite=True)

    pf.writeto(workdir + '/' + 'Smom_0.fits', SSmom_0, head1, overwrite=True)
    pf.writeto(workdir + '/' + 'Smom_1.fits', SSmom_1, head2, overwrite=True)
    pf.writeto(workdir + '/' + 'Smom_2.fits', SSmom_2, head2, overwrite=True)
    pf.writeto(workdir + '/' + 'Smom_8.fits', SSmom_8, head2, overwrite=True)

    pf.writeto(workdir + '/' + 'im_Ipeak.fits',
               SSIpeak,
               headim,
               overwrite=True)

    pf.writeto(workdir + '/' + 'fiterrormap.fits',
               fiterrormap,
               headim,
               overwrite=True)

    if StoreModel:
        pf.writeto(workdir + '/' + 'modelcube.fits',
                   modelcube,
                   headcube,
                   overwrite=True)

    end_time = time.time()
    print("exec_Gfit done in (elapsed time):", end_time - start_time)
