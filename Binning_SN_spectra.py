# Author:   Valentin Reichel
# Date  :   10.05.2023
# Python:   Python 3.8 on Windows
# Subject:  S/N Binning for long-slit-spectoscropy data
########################################################################################################################
import numpy as np

from numpy import sqrt, pi, exp, log, polyfit

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

import warnings
# Suppress rank warning
warnings.filterwarnings("ignore", category=np.RankWarning)
####################################################
def calc_signal_noise(lam, flux):
    '''
    Calculate the signal-to-noise ratio of the galaxy spectra.

    Parameters:
    - lam (array-like): Array containing the wavelength values of the galaxy spectra.
    - flux (array-like): Array containing the flux values of the galaxy spectra.

    Returns:
    - signal_noise (float): The signal-to-noise ratio of the galaxy spectra.
    '''
    # cut out part of the spectrum without/few absorption features
    # wavelength array should be logarithmically rebinned
    lam_min = 8.63923   # section is about 91 Angstroem wide
    lam_max = 8.65525
    flux_section = flux[(lam >= lam_min) & (lam <= lam_max)]
    lam_section = lam[(lam >= lam_min) & (lam <= lam_max)]
    # fit polynom to the data in this part
    pol_coeff = np.polyfit(lam_section, flux_section, 5)
    pol_val = np.polyval(pol_coeff, lam_section)

    # calculate the rms error
    rms_error = np.std(pol_val - flux_section)
    signal_noise = np.median(flux_section / rms_error)
    return signal_noise

# 2.) data access
# read in 2D spectral data, which is conventionally stored in FITS format
dir = r"C:\Users\reich\OneDrive\Dokumente\Uni\Bachelorarbeit\Python\ppxf\spectra"
file_gal = dir + '/ngc_4697_major_axis.fits'  # galaxy spectra along major axis from ngc 4697
# header data unit (HDUs) list are the highest level component of the FITS file structure
# hdu.info() summarizes content of opened fits file
hdu_gal = fits.open(file_gal)
# HDU objects consist of a header and a data unit/attributes
# to read the entire header one can use print(repr(hdr))
hdr_gal = hdu_gal[0].header

data_gal = hdu_gal[0].data
flux_gal_center = data_gal[596, :]  # first look on spectra in SAOImage ds9, row 597 corresponds to a central region
w_gal = WCS(hdr_gal, naxis=1, relax=False, fix=False)
loglam_gal = w_gal.wcs_pix2world(np.arange(len(flux_gal_center)), 0)[0]

# perform binning of data_gal in spatial direction, shape(data_gal)=(1125, 2058)
center_y = 596
sn_threshold = 53

# binning upwards
y_upwards = range(595, -1, -1)
bin_counter = 1  # Initialize the bin counter
for i, row in enumerate(y_upwards):
    flux_row = data_gal[row, :]
    #print("up: Range:", row, "to", center_y )
    flux_bin = np.mean(data_gal[row:center_y, :], axis=0)
    snr = calc_signal_noise(loglam_gal, flux_bin)
    if snr > sn_threshold:
        row_1 = row
        row_2 = center_y
        rad = 0.2 * np.abs((596 - np.mean(np.asarray([row_1, row_2]))))
        print("Bin up: Range:", row, "to", center_y, "Radius:", rad)
        rad_arr = np.full_like(loglam_gal, rad)
        snr_arr = np.full_like(loglam_gal, snr)
        #np.savetxt(f'C:/Users/reich/OneDrive/Dokumente/Uni/Bachelorarbeit/Python/losvds/{bin_counter}_upwards.txt', (loglam_gal, flux_bin, rad_arr, snr_arr))
        center_y = row
        bin_counter += 1
    else:
        continue

# Binning downwards
y_downwards = range(597, data_gal.shape[0], +1)
bin_counter = 1  # Reset the bin counter
center_y = 596   # reset center line y-value
for i, row in enumerate(y_downwards):
    flux_row = data_gal[row, :]
    #print("down: Range:", center_y+1, "to", row + 1)
    flux_bin = np.mean(data_gal[center_y+1:row + 1, :], axis=0)
    snr = calc_signal_noise(loglam_gal, flux_bin)
    if snr > sn_threshold:
        print(center_y)
        row_1 = center_y +1
        row_2 = row+1
        #print(row_1, row_2)
        rad = 0.2 * np.abs((596 - np.mean(np.asarray([row_1, row_2]))))
        print("Bin down: Range:", row_1, "to", row_2, "Radius:", rad)
        rad_arr = np.full_like(loglam_gal, rad)
        snr_arr = np.full_like(loglam_gal, snr)
        #np.savetxt(f'C:/Users/reich/OneDrive/Dokumente/Uni/Bachelorarbeit/Python/bins2/{bin_counter}_downwards.txt',(loglam_gal, flux_bin, rad_arr, snr_arr))
        center_y = row
        bin_counter += 1 # increment bin counter

    else:
        continue
