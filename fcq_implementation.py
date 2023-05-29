# Author:   Valentin Reichel
# Date  :   10.05.2023
# Python:   Python 3.8 on Windows
# Subject:  Implementation of the FCQ algorithm, theoretically described in Bender (1990), to derive
#           LOSVDs from absortpion line spectra and fit a Gauß-Hermite parametrization to them
#############################################################################################
import numpy as np
import scipy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FixedFormatter, FixedLocator

from numpy.fft import fft as fourier, ifft as ifourier
from numpy import sqrt, pi, exp, log, polyfit
from numpy import genfromtxt

from scipy.optimize import curve_fit
from scipy import signal
from scipy.integrate import simps
from scipy.fft import fft, ifft
from scipy.ndimage.filters import convolve
from scipy.signal import correlate
from scipy.interpolate import CubicSpline

import ppxf as ppxf_package
from ppxf.ppxf import ppxf

import warnings
# Suppress rank warning
warnings.filterwarnings("ignore", category=np.RankWarning)
#########################################################################################
# general functions
def remove_continuum_1(loglam_gal, flux_gal, polyord):
    """
    Removes the continuum from an absorption spectrum.

    Parameters:
    - loglam_gal (array-like): Array containing the logarithmically rebinned wavelengths of the spectrum.
    - flux_gal (array-like): Array containing the logarithmically rebinned fluxes of the spectrum.
    - polyord (int): The degree of the polynomial used for fitting.

    Returns:
    - flux_wo_continuum (array-like): The continuum-removed galaxy spectrum
    - poly (array-like): The polynomial fit representing the continuum.
    """
    sig = 3
    deg = polyord

    temp_spectrum = list(np.asarray(flux_gal) + 1.0)
    temp_wl = list(loglam_gal)
    temp_list = [np.asarray([2 for i in range(deg + 1)])]

    """
    The main idea of the algorithm is to iteratively identify and remove the upper fringe of the spectrum by excluding the lower fringe. 
    This is done by following these steps:

    1.)Exclude the lower fringe of the spectrum by defining a threshold value (lowex) based on the standard deviation of the differences between the spectrum and the polynomial fit.
    2.)Fit a polynomial curve to the remaining upper fringe of the spectrum.
    3.)Subtract the polynomial fit from the original spectrum, effectively removing the continuum.
    
    By repeating these steps iteratively, the algorithm progressively refines the polynomial fit and removes the continuum until convergence is achieved.
    """

    while True:
        # Fit a polynomial of degree 'deg' to the current spectrum
        fit = np.polyfit(temp_wl, temp_spectrum, deg)
        poly = np.poly1d(fit)(temp_wl)  # Generate the polynomial fit using the coefficients
        diffs = np.asarray(temp_spectrum) - poly  # Calculate the differences between the spectrum and the polynomial fit

        # Calculate the threshold for excluding data points from the lower fringe of the spectrum
        lowex = np.std(diffs) * sig

        temp_list.append(fit)
        ignorelist = []
        # Identify the data points below the threshold and store their indices
        for i in range(len(diffs)):
            if diffs[i] < -lowex:
                ignorelist.append(i)

        # Remove the identified data points from the spectrum and corresponding wavelengths
        for i in range(len(ignorelist)):
            del temp_spectrum[ignorelist[i] - i]
            del temp_wl[ignorelist[i] - i]

        # Calculate the relative change in the last two polynomial fits as a measure of convergence
        change = sum(abs(temp_list[-1] - temp_list[-2]) / abs(temp_list[-2])) / len(temp_list[-2])
        if change < 0.05:
            break

    # Fit a polynomial of degree 'deg' to the final spectrum
    fitty = np.polyfit(temp_wl, temp_spectrum, deg)
    poly = np.asarray(np.poly1d(fitty)(loglam_gal))  # Generate the polynomial fit using the wavelengths
    flux_wo_continuum = (np.asarray(flux_gal) + 1.0) / poly - 1.0  # Remove the continuum by dividing the spectrum by the polynomial fit and subtracting 1.0
    return flux_wo_continuum, poly

def remove_continuum(gal_flux, gal_wave, temp_flux, temp_wave, gal_sigma, max_iter=10):
    """
    Remove continuum from an elliptical galaxy spectrum using a template star this function uses a bit of a different
    approach then remove_continuum_1.

    Parameters:
    ----------
    gal_flux : array_like
        Flux values of the galaxy spectrum (logarithmically rebinned).
    gal_wave : array_like
        Wavelength values of the galaxy spectrum (logarithmically rebinned).
    temp_flux : array_like
        Flux values of the template star spectrum (logarithmically rebinned).
    temp_wave : array_like
        Wavelength values of the template star spectrum (logarithmically rebinned).
    gal_sigma : float
        Velocity dispersion of the galaxy in km/s.
    max_iter : int, optional
        Maximum number of iterations to perform (default is 10).

    Returns:
    -------
    galaxy_no_continuum : array_like
        The galaxy spectrum with the continuum removed.
    """

    # Define a function to fit a polynomial to the spectrum
    def fit_func(x, *coeffs):
        y = np.zeros_like(x)
        for i, c in enumerate(coeffs):
            y += c * x ** i
        return y

    # Define an initial redshift and velocity dispersion
    redshift = 0.0
    c = 299792.458  # speed of light in km/s
    vdisp = gal_sigma / np.sqrt(2)  # convert sigma to FWHM in km/s

    # Iterate until convergence is reached
    for i in range(max_iter):
        # Apply redshift to galaxy spectrum
        gal_wave_shifted = gal_wave * (1 + redshift)

        # Convolve template star spectrum with Gaussian kernel to match velocity dispersion
        sigma_temp = temp_wave[1] - temp_wave[0]  # assume constant wavelength spacing
        sigma_gal = sigma_temp * vdisp / (c * (1 + redshift))  # convert to km/s at galaxy redshift
        temp_flux_conv = convolve(temp_flux,
                                  np.exp(-(temp_wave / temp_wave[-1]) ** 2 / (2 * (sigma_gal / sigma_temp) ** 2)),
                                  boundary='extend')

        # Interpolate template star spectrum onto galaxy wavelength grid
        temp_flux_interp = np.interp(gal_wave_shifted, temp_wave, temp_flux_conv)

        # Fit a polynomial to the galaxy and template spectra
        p0 = np.ones(4)  # initial guess for polynomial coefficients
        popt, pcov = curve_fit(fit_func, gal_flux / temp_flux_interp, gal_flux, p0=p0)

        # Remove the continuum from the galaxy spectrum
        galaxy_no_continuum = gal_flux - fit_func(gal_flux / temp_flux_interp, *popt) * temp_flux_interp

        # Calculate the new redshift and velocity dispersion
        p0 = [redshift, vdisp]
        popt, pcov = curve_fit(fit_func, gal_wave, galaxy_no_continuum, p0=p0)
        redshift, vdisp = popt

    return galaxy_no_continuum

def calc_redshift(vel_corr, corr_gal_tem):
    """
    Calculates the redshift of a galaxy by analyzing the shifted peaks in the correlation function.

    Parameters:
    - vel_corr (array-like): Array containing the velocity values.
    - corr_gal_tem (array-like): Array containing the correlation values between the galaxy and template.

    Returns:
    - z (float): The redshift of the galaxy.
    - delta_loglam (float): The shift in logarithmic wavelength.
    """
    c = 299792.458  # Speed of light in km/s
    vel_rad = vel_corr[np.argmax(corr_gal_tem)]  # Velocity at the peak of the correlation function between galaxy and template
    delta_loglam = np.log(1 + (vel_rad/c))  # Shift in logarithmic wavelength
    z = np.exp(delta_loglam) - 1  # Redshift
    return z, delta_loglam

def losvd_param(v, amp, v_mean, v_disp, h3, h4):
    """
    Generate a Line-of-Sight Velocity Distribution (LOSVD) based on a Gauß-Hermite-Parametrization based
    on van der Marel & Franx (1993).

    Parameters:
    - v (array-like): Array containing velocity values.
    - amp (float): Amplitude parameter for the LOSVD.
    - v_mean (float): Mean streaming velocity parameter for the LOSVD.
    - v_disp (float): Velocity dispersion parameter for the LOSVD.
    - h3 (float): Third-order Gauss-Hermite moment parameter for the LOSVD.
    - h4 (float): Fourth-order Gauss-Hermite moment parameter for the LOSVD.

    Returns:
    - losvd (array-like): The LOSVD normalized by dividing it by the sum of its elements.
    """

    # Define new variable y for compact notation
    y = np.asarray((np.asarray(v) - v_mean) / v_disp)

    # Compute LOSVD using Gauß-Hermite parametrization
    gaussian_term = np.exp(-0.5 * y**2)
    h3_term = h3 * ((2 * np.sqrt(2) * y**3 - 3 * np.sqrt(2) * y) / np.sqrt(6))
    h4_term = h4 * ((4 * y**4 - 12 * y**2 + 3) / np.sqrt(24))
    losvd = amp * (gaussian_term * (1 + h3_term + h4_term))

    return losvd

def fit_losvd(velocity_corr_peak, broadening_func):
    """
    Fit a Gauss-Hermite parametrization to the line-of-sight velocity distribution (LOSVD).

    Parameters:
    - velocity_corr_peak (array-like): Array containing the velocity values at the peak.
    - broadening_func (array-like): Array containing the broadening function (LOSVD).

    Returns:
    - fitted_parameters (array-like): Array containing the fitted parameters for the LOSVD.
    """

    # Initial parameter values for the fit
    initial_parameters = [175, 1300, 250, 0, 0]

    # Lower bounds for the fit parameters
    lower_bounds = [0, 1000, 100, -0.3, -0.3]

    # Upper bounds for the fit parameters
    upper_bounds = [10 ** 6, 1400, 300, 0.3, 0.3]

    # Perform the curve fit using the Gauss-Hermite parametrization
    fitted_parameters = curve_fit(losvd_param, velocity_corr_peak, broadening_func, p0=initial_parameters,
                                  bounds=(lower_bounds, upper_bounds))[0]

    return fitted_parameters

def signal_model(s, a0, a2):
    """
    Returns the model for the logarithmic power spectrum signal part, which represents a parabola.
    This corresponds to a gaussian signal model in non-logarithmic space.

    Parameters:
    - s (array-like): Input value(s) representing the frequencies (or logarithmic frequencies).
    - a0 (float): Coefficient of the constant term in the parabola equation.
    - a2 (float): Coefficient of the quadratic term in the parabola equation.

    Returns:
    - signal (float or array-like): The computed value(s) of the power spectrum signal.
    """
    return a0 + a2 * np.power(s, 2)


def noise_model(s, b0, b1):
    """
    Returns the model for the logarithmic power spectrum noise part, which represents a straight line.
    This corresponds to an exponential noise model in non-logarithmic space.

    Parameters:
    - s (array-like): Input value(s) representing the frequencies (or logarithmic frequencies).
    - b0 (float): Coefficient of the constant term in the line equation.
    - b1 (float): Coefficient of the linear term in the line equation.

    Returns:
    - noise (float or array-like): The computed value(s) of the power spectrum noise.
    """
    return b0 + b1 * s

########################################################################################################################
def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

# Using seaborn's style
plt.style.use('seaborn-bright')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathrsfs}')
#print(plt.style.available)
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx}",
    "font.family": "serif",
    # Use 10.95pt font in plots, to match 10.95pt font in figure caption
    "axes.labelsize": 10.95,
    "font.size": 10.95,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10.95,
    "xtick.labelsize": 10.95,
    "ytick.labelsize": 10.95,
    # adjust the linewidth of the axes and the grid
    "axes.linewidth":10.95/14,
    "grid.linewidth":0.1
}
plt.rcParams.update(tex_fonts)
########################################################################################################################
# The following 3 lines have to be adapted, here the template data should be loaded in.
# For loading a fits file one can use the astropy fits,
# the flux and wavelength arrays have to be logarithmically rebinned
temp_data = np.loadtxt('C:/Users/reich/OneDrive/Dokumente/Uni/Bachelorarbeit/Python/bins/template.txt')
loglam_temp = temp_data[0,:]
flux_temp = temp_data[1,:]

def fcq():

    def fcq_application(loglam_temp, flux_temp, loglam_gal, flux_gal):
        """
        Returns the model for the logarithmic power spectrum noise part, which represents a straight line.
        This corresponds to an exponential noise model in non-logarithmic space.

        Parameters:
        - loglam_temp (array-like): Logarithmically rebinned wavelength values for the template spectrum.
        - flux_temp (array-like): Logarithmically rebinned fluxes for the template spectrum.
        - loglam_gal (array-like): Logarithmically rebinned wavelength values for the galaxy spectrum.
        - flux_gal(array-like): Logarithmically rebinned fluxes for the galaxy spectrum. S/N per pixel should be 50

        Returns:
        - gh_moments (array-like): The fitted Gauß-Hermite moments.
        - z (float-like): redshift
        """

        # settings for taper strengths
        spec_fourier_taper = 7      # taper before FT for fluxes
        peak_fourier_taper = 7      # taper before FT for correlation function peak
        # settings for polynom degrees for the continuum of the spectra
        polygrad_gal = 8
        polygrad_temp = 8
        # width for the correlation peak
        cut_off_pixel_2 = 65

        # obtain velocity scale per pixel for velocity vector
        c = 299792.458  # speed of light in km/s
        frac = loglam_gal[1] - loglam_gal[0]  # Constant lambda fraction per pixel
        velscale = c * frac
        #vel = np.asarray([-x * velscale for x in range(1, 20)][::-1] + [0] + [x * velscale for x in range(1, 20)])

        #1.) cut off pixels at both ends off arrays before removal of continuum
        # subtract 1 to get uneven number of pixels
        cut_off_pixel = 90
        flux_temp = flux_temp[cut_off_pixel:-(cut_off_pixel-1)]
        flux_gal = flux_gal[cut_off_pixel:-(cut_off_pixel-1)]
        loglam_gal = loglam_gal[cut_off_pixel:-(cut_off_pixel-1)]
        loglam_temp = loglam_temp[cut_off_pixel:-(cut_off_pixel-1)]

        #2.) prepare spectra for fourier analysis, remove continuum level of spectra
        flux_gal /= np.median(flux_gal, 0)
        flux_temp /= np.median(flux_temp, 0)
        flux_temp, cont_poly_temp = remove_continuum_1(loglam_temp, flux_temp, polygrad_temp)
        flux_gal, cont_poly_gal = remove_continuum_1(loglam_gal, flux_gal, polygrad_gal)

        #3.) fourier transform of spectra
        # taper spectrum at both ends using window function np.kaiser()
        # always taper before fourier transform to avoid sharp edges
        taper = np.kaiser(len(flux_gal), spec_fourier_taper)
        flux_temp = flux_temp * taper
        flux_gal = flux_gal * taper
        # fourier transform of spectra (not needed)
        flux_temp_fourier = fourier(flux_temp)
        flux_gal_fourier = fourier(flux_gal)

        #4.) create velocity array
        #[::-1] read array from back to front, e.g. [1,2,3,4][::-1] = [4,3,2,1]
        vel_corr = [-x*velscale for x in range(1, int(((flux_gal.size+1) / 2)))][::-1] + [0] + [x*velscale for x in range(1, int((flux_gal.size+1) / 2))]

        #5.) compute template autocorrelation function and galaxy-template cross correlation function
        # correlation functions
        corr_gal_tem = correlate(flux_gal, flux_temp, mode='same')
        corr_tem_tem = correlate(flux_temp, flux_temp, mode='same')

        #6.)taper correlation functions before FT (not needed since they are not used further)
        #corr_taper = np.kaiser(len(corr_gal_tem), corr_fourier_taper)
        #corr_gal_tem = corr_gal_tem * corr_taper
        #corr_tem_tem = corr_tem_tem * corr_taper

        #7.) extract redshift and shift galaxy wavelength array to rest wavelength
        z, delta_loglam = calc_redshift(vel_corr, corr_gal_tem)

        #8.) extract peak of the correlation functions and velocity array
        # goal of FCQ: deconvolution of correlation function peaks
        peak_index_ccf = np.argmax(corr_gal_tem)
        peak_index_acf = np.argmax(corr_tem_tem)
        # subtract -1 to get odd number of pixels
        corr_gal_tem_peak = corr_gal_tem[peak_index_ccf-cut_off_pixel_2-1:peak_index_ccf+cut_off_pixel_2]
        corr_tem_tem_peak = corr_tem_tem[peak_index_acf-cut_off_pixel_2-1:peak_index_acf+cut_off_pixel_2]
        vel_corr_peak = vel_corr[peak_index_ccf-cut_off_pixel_2-1:peak_index_ccf+cut_off_pixel_2]

        #9.) fourier transform of the correlation peaks
        # taper the correlation peaks with np.kaiser() window before fouriertransform
        corr_peak_taper = np.kaiser(len(corr_gal_tem_peak), peak_fourier_taper)
        corr_gal_tem_peak = corr_gal_tem_peak * corr_peak_taper
        corr_tem_tem_peak = corr_tem_tem_peak * corr_peak_taper
        # fourier transform of correlation functions (not needed)
        corr_gal_tem_fourier = fourier(corr_gal_tem)
        corr_tem_tem_fourier = fourier(corr_tem_tem)
        # fourier transform of correlation function peaks
        corr_gal_tem_peak_fourier = np.fft.fftshift(fourier(corr_gal_tem_peak))
        corr_tem_tem_peak_fourier = np.fft.fftshift(fourier(corr_tem_tem_peak))
        # power spectra in fourier space
        powspec_gal_tem = np.power(np.abs(corr_gal_tem_peak_fourier), 2)
        powspec_tem_tem = np.power(np.abs(corr_tem_tem_peak_fourier), 2)
        # construct x-values in fourier space using np.fft.fftfreq()
        s = np.fft.fftshift(np.fft.fftfreq(corr_gal_tem_peak.size, d=velscale))

        #10.) construct optimal Wiener filter W(s) in fourierspace
        # W(s) = P_model_signal(s)/(P_model_signal(s) + P_model_noise(s)
        # log(powerspec(ccf_peak)) is approximately of the form: / + ∩ + \
        # decompose log(powspec) into signal and noise part for wiener filter

        #cut of frequency where linear model (noise) changes into parabolic model (signal)
        s_inter = 0.0028

        # extract the signal part
        s_signal = s[(s >= -s_inter) & (s <= s_inter)]
        s_signal_left = s_signal[(s_signal <=0)]
        s_signal_right = s_signal[(s_signal >0)]
        log_powspec_signal = np.log(powspec_gal_tem)[(s >= -s_inter) & (s <= s_inter)]

        # extract the noise parts
        s_noise_left = s[(s < -s_inter)]
        log_powspec_noise_left = np.log(powspec_gal_tem)[(s < -s_inter)]
        s_noise_right = s[(s > s_inter)]
        log_powspec_noise_right = np.log(powspec_gal_tem)[(s > s_inter)]

        # fit model of powerspectrum of the signal and noise to observed powerspectrum in fourierspace
        # use log(powerspectrum) -> model signal = parabola, model noise = straight
        signal_model_params = curve_fit(signal_model, s_signal, log_powspec_signal)[0]
        signal_model_vals = signal_model(s_signal, *signal_model_params)
        noise_model_left_params = curve_fit(noise_model, s_noise_left, log_powspec_noise_left)[0]
        noise_model_left_vals = noise_model(s_noise_left, *noise_model_left_params)
        noise_model_right_params = curve_fit(noise_model, s_noise_right, log_powspec_noise_right)[0]
        noise_model_right_vals = noise_model(s_noise_right, *noise_model_right_params)

        # extrapolate values of signal to left and right parts
        signal_model_left = signal_model(s_noise_left, *signal_model_params)
        signal_model_right = signal_model(s_noise_right, *signal_model_params)
        signal_model_all = np.concatenate((signal_model_left, signal_model_vals, signal_model_right), axis=0)
        # extrapolate values of the noise to the center
        noise_model_center_left = noise_model(s_signal_left, *noise_model_left_params)
        noise_model_center_right = noise_model(s_signal_right, *noise_model_right_params)
        noise_model_all = np.concatenate((noise_model_left_vals, noise_model_center_left, noise_model_center_right, noise_model_right_vals), axis=0)
        # add signal and noise in non logarithmic space
        P_signal = np.exp(signal_model_all)
        P_noise = np.exp(noise_model_all)
        P_sum = P_signal + P_noise

        # construct wiener filter: W(k) = P_S(k) /( P_S(k)+P_N(k))
        wiener = P_signal/P_sum

        # 11.) reconstruct broadening function or losvd B(s) in fourier space by applying Wiener filter
        # B(s) = (CCF_peak(s)/ACF_peak(s)) * W(s)

        broadening_func_fourier = np.fft.ifftshift((corr_gal_tem_peak_fourier/corr_tem_tem_peak_fourier)*wiener)
        broadening_func = np.abs(ifourier(broadening_func_fourier))

        #12.) fit gauss hermite parametrization to broadening function

        # fit parametrisation to data using scipy.curve_fit(), least square fitting
        vel_corr_peak = np.array(vel_corr_peak)
        # Create a mask to select values between 600 and 1600 - corresponds to a width of approximately 5 sigma
        mask_vel = (vel_corr_peak >= 1210 - (2.5 * 180)) & (vel_corr_peak <= 1210 + (2.5 * 180))
        # mask = (vel_corr_peak >= np.max(broadening_func) - 500) & (vel_corr_peak <= np.max(broadening_func) + 500)
        gh_moments = fit_losvd(vel_corr_peak[mask_vel], np.fft.fftshift(broadening_func)[mask_vel])

        return gh_moments, z

    def fcq_all_radii():
        """
        Performs the FCQ algorithm for all radii and collects the values of the gh parameters for each radius
        If you want to use this function be sure to use same variable names for spatially binned spectra and adapt the
        bin_number variable.
        """
        c = 299792.458  # speed of light in km/s

        #computes kinematical parameters for all radii
        gal_data_center = np.loadtxt('C:/Users/reich/OneDrive/Dokumente/Uni/Bachelorarbeit/Python/bins/0_central.txt')
        loglam_gal_center = gal_data_center[0, :]
        flux_gal_center = gal_data_center[1, :]
        radius_center = np.asarray(gal_data_center[2, 2])
        signal_to_noise_center = np.asarray(gal_data_center[3, 3])
        gh_moments_center = fcq_application(loglam_temp, flux_temp, loglam_gal_center, flux_gal_center)[0]
        vel_rot_center = gh_moments_center[1]
        vel_disp_center = gh_moments_center[2]
        h3_center = gh_moments_center[3]
        h4_center = gh_moments_center[4]
        z_center = np.asarray(fcq_application(loglam_temp, flux_temp, loglam_gal_center, flux_gal_center)[1])

        bin_number = 46     # number of binned spectra in the upper part
        # generate empty arrays to store the values
        radius_arr = np.empty(bin_number)
        radius_arr[0] = radius_center
        vel_rot_arr = np.empty(bin_number)
        vel_rot_arr[0] = 0
        vel_disp_arr = np.empty(bin_number)
        vel_disp_arr[0] = vel_disp_center
        h3_arr = np.empty(bin_number)
        h3_arr[0] = h3_center
        h4_arr = np.empty(bin_number)
        h4_arr[0] = h4_center
        z_arr = np.empty(bin_number)
        z_arr[0] = z_center
        signal_to_noise_arr = np.empty(bin_number)
        signal_to_noise_arr[0] = signal_to_noise_center

        # perform the fcq algorithm for every spatially binned spectra seperately and store the gh values
        for i in range (1, 46):
            # here the data for the galaxy should be read in, you need to perform the program Binning_SN_spectra.py
            # before this part to achieve approximately constant S/N for every spectrum
            gal_data = np.loadtxt('C:/Users/reich/OneDrive/Dokumente/Uni/Bachelorarbeit/Python/bins2/' + str(i) + '_downwards.txt')
            loglam_gal = gal_data[0, :]
            flux_gal = gal_data[1, :]
            gh_moments, z, *_ = fcq_application(loglam_temp, flux_temp, loglam_gal, flux_gal)

            radius = gal_data[2, 2]
            radius_arr[i] = radius
            vel_rot = np.abs(vel_rot_center-gh_moments[1])
            vel_rot_arr[i] = vel_rot
            vel_disp = gh_moments[2]
            vel_disp_arr[i] = vel_disp
            h3 = gh_moments[3]
            h3_arr[i] = h3
            h4 = gh_moments[4]
            h4_arr[i] = h4
            z_arr[i] = z
            signal_to_noise = gal_data[3, 3]
            signal_to_noise_arr[i] = signal_to_noise

        return radius_arr, vel_rot_arr, vel_disp_arr, h3_arr, h4_arr, z_arr, signal_to_noise_arr

    def fcq_all_radii1():
        """
        Not required it is the same function again, but now just for the lower parts.
        Performs the FCQ algorithm for all radii and collects the values of the gh parameters for each radius
        If you want to use this function be sure to use same variable names for spatially binned spectra and adapt the
        bin_number variable.
        """
        c = 299792.458  # speed of light in km/s

        #computes kinematical parameters for all radii
        gal_data_center = np.loadtxt('C:/Users/reich/OneDrive/Dokumente/Uni/Bachelorarbeit/Python/bins/0_central.txt')
        loglam_gal_center = gal_data_center[0, :]
        flux_gal_center = gal_data_center[1, :]
        radius_center = np.asarray(gal_data_center[2, 2])
        signal_to_noise_center = np.asarray(gal_data_center[3, 3])
        gh_moments_center = fcq_application(loglam_temp, flux_temp, loglam_gal_center, flux_gal_center)[0]
        vel_rot_center = gh_moments_center[1]
        vel_disp_center = gh_moments_center[2]
        h3_center = gh_moments_center[3]
        h4_center = gh_moments_center[4]
        z_center = np.asarray(fcq_application(loglam_temp, flux_temp, loglam_gal_center, flux_gal_center)[1])

        bin_number = 49         # bin number
        # generate empty arrays to store the values
        radius_arr = np.empty(bin_number)
        radius_arr[0] = radius_center
        vel_rot_arr = np.empty(bin_number)
        vel_rot_arr[0] = 0
        vel_disp_arr = np.empty(bin_number)
        vel_disp_arr[0] = vel_disp_center
        h3_arr = np.empty(bin_number)
        h3_arr[0] = h3_center
        h4_arr = np.empty(bin_number)
        h4_arr[0] = h4_center
        z_arr = np.empty(bin_number)
        z_arr[0] = z_center
        signal_to_noise_arr = np.empty(bin_number)
        signal_to_noise_arr[0] = signal_to_noise_center

        # perform the fcq algorithm for every spatially binned spectra seperately and store the gh values
        for i in range (1, 49):
            gal_data = np.loadtxt('C:/Users/reich/OneDrive/Dokumente/Uni/Bachelorarbeit/Python/bins2/' + str(i) + '_upwards.txt')
            loglam_gal = gal_data[0, :]
            flux_gal = gal_data[1, :]
            gh_moments, z, *_ = fcq_application(loglam_temp, flux_temp, loglam_gal, flux_gal)

            radius = gal_data[2, 2]
            radius_arr[i] = radius
            vel_rot = np.abs(vel_rot_center-gh_moments[1])
            vel_rot_arr[i] = vel_rot
            vel_disp = gh_moments[2]
            vel_disp_arr[i] = vel_disp
            h3 = gh_moments[3]
            h3_arr[i] = h3
            h4 = gh_moments[4]
            h4_arr[i] = h4
            z_arr[i] = z
            signal_to_noise = gal_data[3, 3]
            signal_to_noise_arr[i] = signal_to_noise

        return radius_arr, vel_rot_arr, vel_disp_arr, h3_arr, h4_arr, z_arr, signal_to_noise_arr

    def main():
        """main function"""

        # perform FCQ algorithm for every spectrum for all radii in the upper and lower part of the spectrum
        radius_arr, vel_rot_arr, vel_disp_arr, h3_arr, h4_arr, z_arr, signal_to_noise_arr = fcq_all_radii()
        radius_arr1, vel_rot_arr1, vel_disp_arr1, h3_arr1, h4_arr1, z_arr1, signal_to_noise_arr1 =fcq_all_radii1()

        ################################################################################################################
        # plot of resulting kinematical parameters
        width = 412.56497  # width in latex document in pt
        # If you desire to create a figure narrower than the full textwidth you may use the fraction argument.
        # For example, to create a figure half the width of your document use fraction=0.5
        fig_dim = set_size(width/2, fraction=0.5, subplots=(4, 1))
        fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)

        # marker size
        s_size = 1.5

        color_my_data = 'black'

        marker_up = "^"
        marker_down ="v"

        ax1 = axs[0]
        ax1.scatter(radius_arr, h4_arr, s=s_size, color=color_my_data , marker=marker_down, label='FKQ Ergebnisse dieser Arbeit')
        ax1.scatter(radius_arr1, h4_arr1, s=s_size, color=color_my_data, marker=marker_up)

        ax1.set_title('Gauß-Hermite-Parameter der LOSVDs von NGC 4697 (große Hauptachse)', fontsize=10.95, loc='left')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=10.95/18)
        ax1.set_ylabel(r'$h_4$')
        ax1.set_xlim(-2, 65)
        ax1.set_ylim(-0.1, 0.15)
        ax1.set_xticks([])  # Remove x-ticks
        ax1.set_yticks([-0.05, 0, 0.05, 0.1])
        ax1.tick_params(axis='both', direction='in', pad=5, which='both', bottom=True, top=True, left=True, right=True)
        ax1.xaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.025/2))  # Set minor tick positions
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
        ax1.tick_params(axis='both', direction='in', pad=5)

        # Create the legend
        legend = ax1.legend(loc='upper right', frameon=False, fontsize= 8.95)
        legend.legendHandles[0].set_visible(False)  # Hide the marker for the first label
        legend.get_texts()[0].set_color(color_my_data)


        ax2 = axs[1]
        ax2.scatter(radius_arr, -h3_arr, s=s_size, color=color_my_data, marker=marker_down)
        ax2.scatter(radius_arr1, h3_arr1, s=s_size, color=color_my_data, marker=marker_up)

        ax2.axhline(y=0, color='black', linestyle='--', linewidth=10.95 / 18)
        ax2.set_ylabel(r'$h_3$')
        ax2.set_xlim(-2, 65)
        ax2.set_ylim(-0.25, 0.05)
        ax2.set_xticks([])  # Remove x-ticks
        ax2.set_yticks([-0.2, -0.15, -0.1, -0.05, 0])
        ax2.tick_params(axis='both', direction='in', pad=5, which='both', bottom=True, top=True, left=True, right=True)
        ax2.xaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.025/2))  # Set minor tick positions
        ax2.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
        ax2.tick_params(axis='both', direction='in', pad=5)


        ax3 = axs[2]
        ax3.scatter(radius_arr, vel_disp_arr, s=s_size, color=color_my_data, marker=marker_down)
        ax3.scatter(radius_arr1, vel_disp_arr1, s=s_size, color=color_my_data, marker=marker_up)

        ax3.set_ylabel(r'$\sigma\ \text{in}\ \si[per-mode=symbol]{\kilo\meter\per\second}$')
        ax3.set_xlim(-2, 65)
        ax3.set_ylim(130, 190)
        ax3.set_yticks([140, 160, 180])
        ax3.set_xticks([])  # Remove x-ticks
        ax3.tick_params(axis='both', direction='in', pad=5, which='both', bottom=True, top=True, left=True, right=True)
        ax3.xaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
        ax3.yaxis.set_minor_locator(ticker.MultipleLocator(5))  # Set minor tick positions
        ax3.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
        ax3.tick_params(axis='both', direction='in', pad=5)


        ax4 = axs[3]
        ax4.scatter(radius_arr, vel_rot_arr, s=s_size, color=color_my_data, marker=marker_down)
        ax4.scatter(radius_arr1, vel_rot_arr1, s=s_size, color=color_my_data, marker=marker_up)

        ax4.axhline(y=0, color='black', linestyle='--', linewidth=10.95 / 18)
        ax4.set_ylabel(r'$v_{\text{rot}}\ \text{in}\ \si[per-mode=symbol]{\kilo\meter\per\second}$')
        ax4.set_xlabel(r'$r\ \text{in Bogensekunden}$')
        ax4.set_xlim(-1, 65)
        ax4.set_ylim(-10, 180)
        ax4.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax4.set_yticks([0, 50, 100, 150])
        ax4.tick_params(axis='both', direction='in', pad=5, which='both', bottom=True, top=True, left=True, right=True)
        ax4.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))  # Set minor tick positions
        ax4.xaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
        ax4.yaxis.set_minor_locator(ticker.MultipleLocator(10))  # Set minor tick positions
        ax4.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
        ax4.tick_params(axis='both', direction='in', pad=5)

        # Set the x-limits for other axes based on ax4
        ax1.set_xlim(ax4.get_xlim())
        ax2.set_xlim(ax4.get_xlim())
        ax3.set_xlim(ax4.get_xlim())

        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0, hspace=0)
        #plt.savefig(r'C:\Users\reich\OneDrive\Dokumente\Uni\Bachelorarbeit\Arbeit\Vorlage TeX\plots\gh_params.png',
        #            bbox_inches='tight', dpi=300)

        plt.show()
    main()

########################################################################################################################
if __name__ == '__main__':
    fcq()
