"""A Python-3 script to reduce GTC OSIRIS longslit optical spectra. Currently only supporting R300R and R2500I.

It is designed to work over multiple processors and will require a folder 'alt_redspec/' containing
'objects/', 'standards/', 'residuals/' and 'calib_funcs/'. On the upper level it will place the plots
(made in the alt_plot.py script) of the resultant spectra.

It requires packages: numpy, glob, matplotlib, astropy, scipy, warnings, sys, multiprocessing, time and typing.

It will search for spectra in the current directory sub-folder 'Raw/' in which we expect in subsequent order:
'resolution/programme_ID/observing_block/' where those folders correspond with YOUR values
(e.g. 'Raw/R2500I/GTC54-15ITP/OB0001/'), inside each observing block directory we also expect:
'bias/', 'flat/', 'object/' and 'stds/' inside which are the fits files beginning with '0'.

Required non script files/ folders in the same directory as this script include:
    * Master_info_correct_cm.csv    -- Containing the filenames, programme IDs, observing blocks, resolutions,
                                       shortnames, distances, spectral type and spectral type number.
    * observing_qc_positions_parseable.txt  -- Containing the programme IDs, observing bocks, resolutions
                                               and x pixel where the spectra is on the image (default is 250).
    * calib_models/     -- Containing the WD models that correspond with the observed standards
    * alt_done.log      -- An empty text file to be filled with the files that have been reduced already
    * good_wds.txt      -- A descriptive file of which standards per observing block should be used
    * BPM_(resolution)_python.txt       -- A file for the bad pixel masks for the resolution (else one will be made)
    * alt_doplot.log    -- A descriptive file for which spectra should be plotted per observing block
    * (resolution)_arcmap.txt       -- A file of pixel and wavelength
    * R300R_correction_function.txt     -- Correcting second order contamination in R300R spectra

Classes
-------
OB : str
    The full class, passed the string to the observing block and hence reduces the spectra in that observing block
BPM : str
    Determines the bad pixel mask if the corresponding resolution mask does not exist in the current environment

Methods
-------
repeat_check
    Checks if an object has been observed in multiple observing blocks and median stacks if so
main
    Main function of the script to make bad pixel mask and reduce the spectra
"""
# TODO: Convert the above described requirements into something more soft coded that a user can provide.
# TODO: Create example folder if githubbing so people can make their equivalents
import numpy as np  # general mathematics and array handling
from astropy.io import fits  # opening multiple HDU fits files
from astropy.table import Table  # opening files as data tables
import scipy.interpolate as sinterp  # for interpolation (we use linear)
import matplotlib.pyplot as plt

import argparse
import glob  # equivalent to linux 'ls'
import warnings  # used for suppressing an annoying warning message about runtime
import sys  # system arguments
import multiprocessing  # used to overcome the inherent python GIL
import time  # timing processing
from typing import Tuple  # used for type hinting
import os


class OB:
    """Reduces the optical spectra

    This is the primary class of this script which self constructs using a string UNIX path to the observing block
    which contains the spectra. It will sequentially determine the bias and flat field before applying them
    to firstly the closest WD to the target (in airmass and humidity terms) -- to create a calibration function
    which will give us absolutely calibrated fluxes -- for which we have models. Then the actual standard of that
    observation block, and finally the target.

    The pixel coordinate corresponding to the location of the spectra on the CCD has already been visually found,
    this is used as the centering value of a 50 pixel chunk, in which we find the peak signal. We extract then
    an aperture around that peak with a width of 10 percent of peak. This is summed and the background is subtracted
    (the median of the 50 pixel chunk minus the aperture).

    Wavelength is determined by fitting a 4th order polynomial to the line list (from OSIRIS) to allow us to convert
    pixel. The dispersion axis is along the columns, y axis, of the CCD and the central x pixel tends to be 250
    (or 265 for the R2500I grism due to spatial displacement).

    The calibration function is created by dividing the extracted WD standard by its respective model (in absolute flux,
    i.e. at the Earth); the model is linearly interpolated to become dimensionally equivalent to the observation.
    The extracted target is then divided by the calibration function and multiplied by the factor
    distance greater than 10pc it is squared.

    Attributes
    ----------
    ob : str
        The observing block
    prog : str
        The programme ID
    resolution : str
        The resolution of this spectra
    coordinate : int
        The x coordinate (pixel - 1) of where the spectra is aligned on the slit
    ptobias : str
        The path to the bias files
    master_bias : numpy.ndarray
        Median stacked bias files, full CCD
    ptoflats : str
        The path to the flats used
    master_flat : numpy.ndarray
        Median stacked, bias subtracted and normalised flat field
    ptoobj : str
        The path to the target files
    humidity : float
        The average humidity of the target observations
    airmass : float
        The average airmass of the target observations
    ptostds : str
        The path to the standards observed in the same block
    ftoabs : numpy.ndarray
        The calibration function for the conditions of the night
    master_standard : numpy.ndarray
        The wavelength, flux and flux error of the standard in absolute units
    ftoabs_error : numpy.ndarray
        The error on the calibration function
    standard_name : str
        The standard actually used for the calibration (not neccesarily the standard of the observing block)
    standard_residual : numpy.ndarray
        The cut out region of the CCD around the spectra
    actual_standard : str
        The actual standard from the same observing block
    master_target : numpy.ndarray
        Wavelength, flux and flux error in absolute units of the target
    target : str
        The shortname of the target, e.g. J1234+5678
    target_residual : numpy.ndarray
        The cut out region of the CCD around the spectra

    Methods
    -------
    get_targetname(sname)
        Pulls the name of the target from the header and converts to shortname
    get_pixcoord()
        Finds the coordinate where the spectra is on the CCD
    bisub(data)
        Subtracts the bias from the data array
    bpm_applying(data)
        Multiplies the bad pixel mask by the data array (0 for bad pixels)
    fopener()
        Opens the fits file to the correct hdu and extracts the data
    fopenbisub(fname)
        Opens the fits file and subtracts the bias from the data within
    med_stack(all_data)
        Takes an array of all observations stacked in the 3rd dimension and compresses to a 2D median
    bias()
        Creates the bias file used for the reduction
    normalise(data)
        Normalised the flat field on the median of a select region (nominally around where the spectra tends to be)
    flat()
        Creates the flat field used for the reduction
    transform(pixel)
        Converts pixel values to wavelength after fitting a polynomial to the arc map
    checkifspectra(spectral_list)
        Ascertains that the spectra fits file is not an acquisition (they are in the same directory)
    flat_field(data)
        Divides the spectral CCD by the flat field
    inf_fix(data)
        Effectively performs the bad pixel masking of the spectra CCD (during flat fielding division where bad pixels
        are 0, numpy provides an infinite value, this function resets them to be 0)
    back_subtract(data, back)
        Using a CCD row, subtracts the background of that row from the actual signals
    peak_average(segment)
        Takes a 50 pixel row of the CCD around the spectra, finds the peak and ignores cosmic rays and sums over it
    extract(data)
        Extracts the spectra along the dispersion axis row by row
    poisson(photon_count)
        Determines the Poisson noise on the photon count signal
    calibrate_errorprop(f, errs, errv, v)
        Propagates the error through use of the calibration function
    hum_air()
        Finds mean humidity and airmass of the observations
    humairdiff(line)
        Gets difference between standard airmass and humidity and object
    closest_standard()
        Determines which standard is closest to the object in terms of airmass and humidity
    object(ptoobj)
        Reduces the target spectra
    writing(ptostore)
        Writes the reduced spectra to txt files
    """

    def __init__(self, ptodata: str):
        """
        Parameters
        ----------
        ptodata : str
            The UNIX path as a string to where the observing block is
        """
        tproc0 = time.time()
        self.ob = ptodata.split('/')[-1]  # observing block of this spectra
        self.prog = ptodata.split('/')[2]  # program ID
        self.resolution = ptodata.split('/')[1]  # resolution of this spectra
        if self.target_check():
            self.figobj, self.axesobj = plt.subplots(4, 4, figsize=(16, 12), dpi=300)
            self.axesobj = self.axesobj.flatten()
            self.figstd, self.axesstd = plt.subplots(4, 4, figsize=(16, 12), dpi=300)
            self.axesstd = self.axesstd.flatten()
            self.coordinate = self.get_pixcoord()  # find pixel coordinate
            self.pixlow, self.pixhigh = self.pixel_constraints()
            self.ptobias = ptodata + '/bias/0*fits'  # path to biases
            self.master_bias = self.bias(self.ptobias)  # creates the master bias file
            self.axesobj[4].imshow(self.master_bias, cmap='plasma', origin='lower', aspect='auto')
            self.ptoflats = ptodata + '/flat/0*fits'  # path to flats
            self.master_flat = self.flat(self.ptoflats, self.master_bias)  # creates the master flat file
            self.axesobj[5].imshow(self.master_flat, cmap='plasma', origin='lower', aspect='auto')
            self.bpm = self.bpm_applying(np.ones_like(self.master_flat))
            self.axesobj[6].imshow(self.bpm, cmap='plasma', origin='lower', aspect='auto')
            self.ptoobj = ptodata + '/object/0*fits'  # path to object
            self.humidity, self.airmass, self.mjd = self.hum_air()  # average humidity and airmass of object obs
            self.ptostds = ptodata + '/stds/0*scopy.fits'  # path to standard
            self.ftoabs, self.master_standard, self.ftoabs_error, self.standard_name, \
            self.standard_residual = self.standard()  # reduces the closest standard to target
            self.actual_standard = self.standard_name
            # self.master_standard, self.actual_standard = self.object(self.ptostds)[:2]  # reduces block standard
            self.master_target, self.target, self.target_residual = self.object(self.ptoobj)  # reduces target
            self.writing('alt_redspec')  # writes reduced spectra to files
            for axes in (self.axesobj, self.axesstd):
                axes[1].set_title('Bias Subtracted')
                axes[2].set_title('Flat Fielded')
                axes[3].set_title('Bad Pixel Masked')
                axes[4].set_title('Bias')
                axes[5].set_title('Flat')
                axes[6].set_title('Bad Pixel Mask')
                axes[7].set_title('Extraction Region')
                axes[8].set_title('Extracted Spectra')
                axes[9].set_title('Wavelength Calibrated')
                axes[12].set_title('Arc Map')
            self.axesobj[10].set_title('Final')
            self.axesstd[10].set_title('Wavelength Calibrated')
            self.axesobj[13].set_title('Calibration Function')
            self.axesstd[11].set_title('Final')
            self.axesstd[13].set_title('Model')
            self.axesstd[14].set_title('Calibration Function')
            [self.figobj.delaxes(self.axesobj[i]) for i in [11, 14, 15]]
            [self.figstd.delaxes(self.axesstd[i]) for i in [15, ]]
            self.figobj.savefig(f'alt_redspec/reduction/{self.ob}_{self.resolution}_{self.prog}_{self.target}.png',
                                bbox_inches='tight')
            self.figstd.savefig(f'alt_redspec/reduction/{self.ob}_{self.resolution}_{self.prog}'
                                f'_{self.standard_name}.png',
                                bbox_inches='tight')
            print(f'Object processed: {self.target} for {self.resolution} '
                  f'grism in {self.ob} with walltime {round(time.time() - tproc0, 1)} seconds.')
        else:
            print(f'Object in Observing Block {self.ob} in Programme {self.prog} '
                  f'with Resolution {self.resolution} will not be reduced.')
        return

    def target_check(self) -> bool:
        """Checks if a target observation should be processed

        Opens up the table of sources and compares observing block, programme ID against this source,
        if the object is not in the table, don't waste time reducing it.
        """
        tinfo = Table.read('Master_info_correct_cm.csv')  # table containing observation information
        for row in tinfo:
            if row['OB'].strip() == self.ob and row['Program'].strip() == self.prog \
                    and row['Resolution'].strip() == self.resolution:
                dosource = True
                break
        else:
            dosource = False
        return dosource

    @staticmethod
    def get_targetname(sname: str) -> str:
        """Gets target name from external table

        Compares the header name of the observation to the table, if a match is found, use the corresponding
        shortname as the target name for use henceforth.

        Parameters
        ----------
        sname : str
            The name of the object to be compared with the table
        """
        tinfo = Table.read('Master_info_correct_cm.csv')  # table containing header names and shortnames
        sname = sname.strip()  # remove whitespace
        for row in tinfo:
            if row['Target'].strip() == sname:  # if the header names match
                sname = row['SHORTNAME']  # take the shortname as the target name
                break
        return sname

    def get_pixcoord(self) -> int:
        """Gets the pixel coordinate

        The row pixel where the spectra is should be recorded in the listed file (3 columns, programme ID,
        observation block, and pixel coordinate.
        """
        # the table referenced in the doc string
        t = Table.read('observing_qc_positions_parseable.txt', format='ascii.no_header', names=('prog', 'ob', 'pix'))
        coord = 250  # default coordinate pixel
        for row in t:
            if row['prog'] == self.prog and row['ob'] == self.ob:  # if the row programme ID and observation block match
                coord = row['pix']  # use that pixel
                break
        if self.resolution == 'R2500I':  # if the resolution is R2500I
            coord += 15  # there is spatial displacement worth 15 pixels
        return coord - 1  # pixel to index conversion

    def pixel_constraints(self) -> Tuple[int, int]:
        """Finds the constraints for the respective resolution

        Using the resolution of this object, find the limits on the dispersion axis at which to extract spectra
        """
        if self.resolution == "R2500I":
            xmin = 40
            xmax = 1900
        else:
            xmin = 830
            xmax = 1410
        return xmin, xmax

    @staticmethod
    def bisub(bias: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Bias subtract from each ccd given

        Elementwise subtraction of the entire CCD, pixel by pixel

        Parameters
        ----------
        bias : np.ndarray
            The bias from same observing block as the object
        data : np.ndarray
            The full CCD that will have the bias subtracted
        """
        return data - bias

    def bpm_applying(self, data: np.ndarray) -> np.ndarray:
        """Applies the bad pixel mask

        Elementwise multiplication where 0 is a bad pixel and 1 (unchanged) for good pixels

        Parameters
        ----------
        data : np.ndarray
            The full CCD that needs to be masked
        """
        mask = np.loadtxt(f'BPM_{self.resolution}_python.txt')
        return mask * data  # BPM is array of 1=good, 0=bad

    @staticmethod
    def fopener(fname: str) -> np.ndarray:
        """Opens the fits file using astropy

        Parameters
        ----------
        fname : str
            The relative or full string to the file to be opened
        """
        with fits.open(fname) as hdul:
            data = hdul[2].data  # OSIRIS longslit is targeted on the second CCD
        return data

    def fopenbisub(self, bias: np.ndarray, fname: str) -> np.ndarray:
        """Uses the file opening method and bias subtracting method

        Parameters
        ----------
        bias: np.ndarray
            The bias numpy array
        fname : str
            The relative or full string to the file to be opened
        """
        return self.bisub(bias, self.fopener(fname))  # opens and then bias subtracts the file using the listed methods

    @staticmethod
    def med_stack(all_data: np.ndarray) -> np.ndarray:
        """Median stacks all the CCDs

        If an object is observed more than once in an observing block, this function determines the medians.
        This works by taking the stacked CCDs and determining the median for each pixel (all CCDs will be the same
        size arrays). Physically the shape will be (number of CCDs, number of rows, number of columns).

        Be aware that if called outside of this script to ensure you are passing a 3D array.

        Parameters
        ----------
        all_data : np.ndarray
            A 3D array of all the observation CCDs (2D)

        Raises
        ------
        WrongShapeError
            If given a non 3D array
        """
        if len(all_data.shape) != 3:
            class WrongShapeError(Exception):
                pass

            raise WrongShapeError(f'all_data must be a 3D array not {all_data.shape}')
        return np.median(all_data, axis=0)  # median stacks along the 3rd dimension

    def bias(self, ptobias: str) -> np.ndarray:
        """Creates master bias

        All the biases are taken from the fits files in the UNIX path given in __init__. They are then 3D stacked
        and then unstacked along the median back to a 2D CCD.

        Parameters
        ----------
        ptobias : str
            The path to where the biases are stored
        """
        bias_list = glob.glob(ptobias)  # list all the biases
        all_bias = np.stack([self.fopener(bias) for bias in bias_list])  # creates an array of the bias CCDs
        median_bias = self.med_stack(all_bias)  # median stacks the biases
        return self.bpm_applying(median_bias)  # apply bad pixel mask

    def normalise(self, data: np.ndarray) -> np.ndarray:
        """Normalises to the mean value of array

        Depending on the resolution of this spectra, select the region around where the first order spectra usually is
        and normalise the entire CCD to the mean of that select region. The selected rows have beeen determined
        by inspecting the location of the arc lines.

        Parameters
        ----------
        data : np.ndarray
            The full CCD of the median flat
        """
        if self.resolution == 'R2500I':
            dtrimmed = data[self.pixlow:self.pixhigh, 165:365]
        else:
            dtrimmed = data[self.pixlow:self.pixhigh, 150:350]
        return data / np.median(dtrimmed)

    def flat(self, ptoflats: str, bias: np.ndarray) -> np.ndarray:
        """Creates master flat

        All the flats are taken from the fits files in the UNIX path given in __init__. They are then 3D stacked
        and then unstacked along the median back to a 2D CCD. They then have the bad pixel mask applied and
        are finally normalised to the spectral region.

        Parameters
        ----------
        ptoflats : str
            The string path to where the flats are formed
        bias : np.ndarray
            The full CCD of the bias file
        """
        flat_list = glob.glob(ptoflats)  # list of all flats in observing block
        all_flats = np.stack([self.fopenbisub(bias, flat) for flat in flat_list])
        median_flat = self.med_stack(all_flats)  # determines bias subtracted median flat
        bpm_flat = self.bpm_applying(median_flat)  # apply bad pixel mask
        return self.normalise(bpm_flat)  # normalised flat

    def transform(self, pixel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixels into wavelength

        Loads the given file from OSIRIS, extracts the pixel and wavelength then fits a 3rd order polynomial
        and applies that to all the pixels given.

        Parameters
        ----------
        pixel : np.ndarray
            1D array of pixel values along the dispersion axis
        """
        arcpix, arcwav = np.loadtxt(f'{self.resolution}_arcmap.txt', unpack=True, usecols=(0, 2))  # pixels to wave
        co_eff = np.polyfit(arcpix, arcwav, 3)  # the coefficients of the polynomial from pixel to wavelength
        f = np.poly1d(co_eff)  # the fit to those coefficients
        return f(pixel), arcpix, arcwav  # wavelengths

    @staticmethod
    def checkifspectra(spectral_list: list) -> np.ndarray:
        """Checks if the file is an acquisition file or a real spectra

        This method opens every file in the given list, and looks at the GRISM keyword of the header.
        If the GRISM is 'OPEN' it is an acquisiton image, hence should be ignored.

        Parameters
        ----------
        spectral_list : list
            The list of files that have spectra
        """
        bo_arr = np.array([], dtype=bool)  # empty boolean array
        spectral_list = np.array(spectral_list)  # convert to numpy array
        for spectra in spectral_list:
            with fits.open(spectra) as hdul:
                head = hdul[0].header
                if head['GRISM'] == 'OPEN':  # i.e. acquisiton
                    bo_arr = np.append(bo_arr, False)
                else:
                    bo_arr = np.append(bo_arr, True)
        return spectral_list[bo_arr]  # only use files that are actual spectra

    @staticmethod
    def flat_field(flat: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply the normalised flat field

        Note that as the flat field has had a bad pixel mask over it, division by zero causes the output of this
        method to contain infinite counts where the bad pixel is. Use inf_fix to remedy this.

        Parameters
        ----------
        flat : np.ndarray
            The full CCD of the flat to be used
        data : np.ndarray
            The full CCD that will needs to be flat fielded
        """
        return data / flat

    @staticmethod
    def inf_fix(data: np.ndarray) -> np.ndarray:
        """Where there is a bad pixel, the data will now be infinite. This will set those to 0.

        Parameters
        ----------
        data : np.ndarray
            The flat fielded CCD which contains infinite values
        """
        temp = data.flatten()  # turns the 2D CCD into 1D
        for i, val in enumerate(temp):
            if np.isinf(val):  # if the value is infinite
                temp[i] = 0  # set it to 0 (bad pixel)
        return temp.reshape(data.shape)  # reshape back to 2D, but with bad pixels now 0 not infinite

    @staticmethod
    def back_subtract(data: float, back: float) -> float:
        """Subtracts the background from the extracted spectrum

        Finds the background subtracted signal, if signal is less than the background (a median) set it to 0.

        Parameters
        ----------
        data : float
            The signal that needs to be subtracted
        back : float
            The background value to subtract from the signal
        """
        backsubbed = data - back
        if backsubbed < 0:  # if the signal was less than the background
            backsubbed = 0  # set to 0
        return backsubbed

    @staticmethod
    def recenter(segment: np.ndarray, cpix: int) -> int:
        """Finds the center of the aperture

        Parameters
        ----------
        segment : np.ndarray
            A 1D array of the counts
        cpix : int
            The current central pixel
        """
        # cpix = ind[len(ind) // 2]  # the central index of the row
        cpix = np.argmax(segment[cpix - 10:cpix + 10]) + cpix - 10  # the new central row index where the peak is in 10
        return cpix

    def peak_average(self, segment: np.ndarray, cpix: int) -> Tuple[float, int, int, int]:
        """Takes the strip of the CCD and gets the median around the peak

        This method extracts the full signal. It is given a 50 pixel row centered on the pixel coordinate where the
        spectra is. First it fits a straight line to the signal and takes the mean of that as the background.
        Then, starting with the central value of the row, it finds the peak within 5 pixels of the center (peak signal)
        in order to recenter. After subtracting the background from every value along the full row
        it creates an aperture with a maximum width of 10 pixels or to the boundaries where the count drops below
        10 percent of the peak.

        Parameters
        ----------
        segment : np.ndarray
            The 50 pixel row around the spectral pixel coordinate
        cpix : int
            The central pixel value
        """
        if np.array(segment == np.zeros_like(segment)).all():  # bad pixel rows
            return 0
        ind = np.arange(len(segment))  # an array of indices in the same length as the row
        med = np.median(segment)  # the median of the row
        w = self.inf_fix(1 / abs(med - segment))  # weights determined by distance from the median
        # the closer the count to the median, the higher it is weighted, i.e. real signal and cosmic rays are supressed
        pfit = np.poly1d(np.polyfit(ind, segment, 1, w=w))  # straight line fit weighted by distance from median
        back = pfit(ind)  # values of the straight line fit
        backmean = back.mean()  # the mean straight line value is the background
        cpix = self.recenter(segment, cpix)
        leftwidth, rightwidth = self.aperture_width(segment, cpix)
        minind = cpix - leftwidth
        maxind = cpix + rightwidth
        backsub = np.array([self.back_subtract(i, backmean) for i in segment])  # subtract the background from row
        signal = np.sum(backsub[minind + 1:maxind])  # sum up all the back subtracted signal in aperture
        return signal, minind, cpix, maxind

    def aperture_width(self, segment: np.ndarray, cpix: int) -> Tuple[int, int]:
        """Determines the aperture width to be used for the full extraction

        Takes a given line at approximately 8150A and determines the width of the aperture
        by resizing to 10 percent.

        Parameters
        ----------
        segment : np.ndarray
            The 50 pixel row around the spectral pixel coordinate
        cpix : int
            The central pixel
        """
        ind = np.arange(len(segment))  # an array of indices in the same length as the row
        med = np.median(segment)  # the median of the row
        w = OB.inf_fix(1 / abs(med - segment))  # weights determined by distance from the median
        # the closer the count to the median, the higher it is weighted, i.e. real signal and cosmic rays are supressed
        pfit = np.poly1d(np.polyfit(ind, segment, 1, w=w))  # straight line fit weighted by distance from median
        back = pfit(ind)  # values of the straight line fit
        backmean = back.mean()  # the mean straight line value is the background
        backsub = np.array([self.back_subtract(i, backmean) for i in segment])  # subtract the background from row
        relevel = 0.1 * backsub[cpix - 5:cpix + 6].max()  # 10 percent of the peak signal
        try:
            minind = np.argwhere(backsub[cpix - 5:cpix] < relevel)[-1][0] + cpix - 5  # lower aperture limit
            maxind = np.argwhere(backsub[cpix:cpix + 6] < relevel)[0][0] + cpix  # upper aperture limit
        except IndexError:  # if there is very low signal use the full width aperture
            minind = cpix - 5
            maxind = cpix + 6
        leftwidth = cpix - minind
        rightwidth = maxind - cpix
        return leftwidth, rightwidth

    def extract(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the spectrum

        Take a 50 pixel slice around the central pixel with a sub-section of rows selected from the CCD based on the
        arc map (e.g. to ensure only first order diffraction in R300R). Then extract and background subtract the
        signal from that slice on a row by row basis.

        Parameters
        ----------
        data : np.ndarray
            The full CCD of the observation, to be sliced and extracted from
        """
        if self.resolution == 'R2500I':
            data = data[:, self.coordinate - 25:self.coordinate + 25]  # slicing spectra out
            pixels = np.arange(1, len(data) + 1)
            # dline = data[625].flatten()
            # N.B. 'pixels' are used throughout but the pixel - 1 = python index has been done
        else:
            data = data[self.pixlow:self.pixhigh, self.coordinate - 25:self.coordinate + 25]  # slicing spectra
            pixels = np.arange(self.pixlow, self.pixhigh)
            # dline = data[1175 - self.pixlow].flatten()
        # lw, rw = self.aperture_width(dline)
        peaks, aptleft, aptcent, aptright = np.empty_like(pixels), np.empty_like(pixels),\
            np.empty_like(pixels), np.empty_like(pixels)
        for i, row in enumerate(data):
            if not i:
                cpix = len(row) // 2
            else:
                cpix = aptcent[i - 1]
            peak_extract = self.peak_average(row, cpix)
            peaks[i] = peak_extract[0]
            aptleft[i] = peak_extract[1]
            aptcent[i] = peak_extract[2]
            aptright[i] = peak_extract[3]
        return pixels, peaks, data, aptleft, aptcent, aptright

    @staticmethod
    def poisson(photon_count: np.ndarray) -> np.ndarray:
        """Returns the photon count error as Poisson noise

        sigma = sqrt(N) / N

        Parameters
        ----------
        photon_count : np.ndarray
            The 1D array of all the counts
        """
        return np.sqrt(photon_count) / photon_count

    @staticmethod
    def calibrate_errorprop(f: np.ndarray, errs: np.ndarray, errv: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Propagates the error from division

        S is the signal being divided by V such that F = S / V therefore sigF^2 = (sigs^2 + F^2 sigv^2)/V^2

        Parameters
        ----------
        f : np.ndarray
            The calibration function or object being divided 1D array
        errs : np.ndarray
            The error in the observations 1D array
        errv : np.ndarray
            The error in the object being divided by as a 1D array
        v : np.ndarray
            The 1D array of the object being divided by
        """
        top = (errs ** 2) + ((f ** 2) * (errv ** 2))
        bottom = v ** 2
        return np.sqrt(top / bottom)

    @staticmethod
    def confining_region(wave: np.ndarray, flux: np.ndarray,
                         error: np.ndarray, wave_check: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Confines the wave and flux arrays to the same regime as the wave_check limits

        Parameters
        ----------
        wave : np.ndarray
            The 1D array of wavelengths to be potentially truncated
        flux : np.ndarray
            The 1D array of fluxes or counts to be potentially truncated
        error : np.ndarray
            The 1D array error on the fluxes or counts to be potentially truncated
        wave_check : np.ndarray
            The 1D array of wavelengths against which the target wave will be compared
        """
        flux = flux[np.logical_and(wave >= np.min(wave_check), wave <= np.max(wave_check))]
        error = error[np.logical_and(wave >= np.min(wave_check), wave <= np.max(wave_check))]
        wave = wave[np.logical_and(wave >= np.min(wave_check), wave <= np.max(wave_check))]
        return wave, flux, error

    def vector_func(self, whichstd: str, wave: np.ndarray,
                    flux: np.ndarray, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divides the standard by a model to get a vector constant

        Loads the model that calibrates with the best WD standard for these objects, interpolates them down to
        the resolution of the standard spectra. Then finds the calibration function and propagates the error
        into that (error determined from the models).

        Parameters
        ----------
        whichstd : str
            A string which is the name of the WD that has standard observations and a calibration model
        wave : np.ndarray
            A 1D array of wavelengths of the WD standard
        flux : np.ndarray
            A 1D array of photon counts of the WD standard
        error : np.ndarray
            A 1D array of errors on the WD standard counts
        """
        vegwave, vegflux = np.loadtxt(f'calib_models/{whichstd}_mod.txt', unpack=True)  # load the model
        vegerr = np.ones_like(vegwave) * (np.std(vegflux) / len(vegflux))  # determine the error
        fluxfunction = sinterp.interp1d(vegwave, vegflux)  # linearly interpolate wavelength to flux
        errfunction = sinterp.interp1d(vegwave, vegerr)  # linearly interpolate wavelength to flux error
        wave, flux, error = self.confining_region(wave, flux, error, vegwave)  # constrain observation to the model
        vegfluxpre, vegerrpre = vegflux.copy(), vegerr.copy()
        vegflux = fluxfunction(wave)  # find the model fluxes from the interpolation at observation resolution
        vegerr = errfunction(wave)  # find the model errors from the interpolation at observation resolution
        ftoabs = flux / vegflux  # the calibration function as counts to absolute flux
        comb_error = self.calibrate_errorprop(ftoabs, error, vegerr, vegflux)  # propagate the error into function
        return wave, flux, error, ftoabs, comb_error, vegwave, vegfluxpre, vegerrpre

    @staticmethod
    def get_header_info(fname: str) -> Tuple[str, float, float, float]:
        """Gets the standard file name

        Parameters
        ----------
        fname : str
            The UNIX path to the file being opened as a string
        """
        with fits.open(fname) as hdul:
            head = hdul[0].header  # the observational information on OSIRIS is on the first HDU
        return head['OBJECT'].rstrip(), head['HUMIDITY'], head['AIRMASS'], head['MJD-OBS']

    def standard(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        """Reduces standard and creates conversion to absolute units

        This method finds the closest standards observation block to the observation in terms of airmass and humidity
        before loading that observation block (note. that is not neccesarily the same observation block as the object).
        The name of that WD standard is used to open the corresponding WD model. The observation is stacked on its
        median, bias subtracted, bad pixel masked and flat fielded before the signal is extracted along
        the dispersion axis with pixels converted to wavelength. Using the model, a calibration function of counts
        to absolute flux (at the Earth) is determined.
        """
        best_std = self.closest_standard().split('_')
        res, prog, obsblock = best_std[2], best_std[1], best_std[0]
        ptobest_std = f'Raw/{res}/{prog}/{obsblock}/stds/0*scopy.fits'
        ptobest_bias = f'Raw/{res}/{prog}/{obsblock}/bias/0*.fits'
        bias = self.bias(ptobest_bias)
        self.axesstd[4].imshow(bias, cmap='plasma', origin='lower', aspect='auto')
        ptobest_flat = f'Raw/{res}/{prog}/{obsblock}/flat/0*.fits'
        flat = self.flat(ptobest_flat, bias)
        self.axesstd[5].imshow(flat, cmap='plasma', origin='lower', aspect='auto')
        self.axesstd[6].imshow(self.bpm, cmap='plasma', origin='lower', aspect='auto')
        standard_list = self.checkifspectra(glob.glob(ptobest_std))  # list of standards)
        sname = self.get_header_info(standard_list[-1])[0]  # gets name of standard
        sname = sname.split('_')[-1].lower().replace('-', '')  # converting standard name from header to model name
        all_standards = np.stack([self.fopener(obj) for obj in standard_list])
        median_standard = self.med_stack(all_standards)  # median stack objects
        self.axesstd[0].imshow(median_standard, cmap='cividis', origin='lower', aspect='auto')
        self.axesstd[0].set_title(f'Median Stack ({len(all_standards)} standard/s)')
        all_standards = np.stack([self.bisub(bias, std) for std in all_standards])
        median_standard = self.med_stack(all_standards)  # median stack the bias subtracted standards
        self.axesstd[1].imshow(median_standard, cmap='cividis', origin='lower', aspect='auto')
        flat_standard = self.flat_field(flat, median_standard)  # flat field the standard
        self.axesstd[2].imshow(flat_standard, cmap='cividis', origin='lower', aspect='auto')
        fixed_standard = self.inf_fix(flat_standard)  # fix infinite counts, acts as a bad pixel mask
        self.axesstd[3].imshow(fixed_standard, cmap='cividis', origin='lower', aspect='auto')
        pixel, photons, resid, aptleft, aptcent, aptright = self.extract(fixed_standard)  # extracts the spectra
        aptleftdiff = aptcent - aptleft
        aptrightdiff = aptright - aptcent
        aptcent = aptcent - 25 + self.coordinate
        aptleft = aptcent - aptleftdiff
        aptright = aptcent + aptrightdiff
        self.axesstd[7].imshow(resid, cmap='coolwarm', origin='lower', aspect='auto',
                               extent=(self.coordinate - 25, self.coordinate + 25, self.pixlow, self.pixhigh))
        self.axesstd[7].plot(aptleft, pixel, color='black', lw=1, ls='--')
        self.axesstd[7].plot(aptright, pixel, color='black', lw=1, ls='--')
        self.axesstd[7].plot(aptcent, pixel, color='black', lw=1, ls='-')
        error = self.poisson(photons)  # creating the errors
        self.axesstd[8].errorbar(pixel, photons, yerr=error)
        wave, arcpix, arcwave = self.transform(pixel)  # transform pixels to wavelength
        self.axesstd[9].errorbar(wave, photons, yerr=error)
        self.axesstd[12].plot(arcpix, arcwave, color='red')
        wave, photons, error, ftoabs, ftoabs_error,\
            vegwave, vegflux, vegerr = self.vector_func(sname, wave, photons, error)  # vector constant
        self.axesstd[10].errorbar(wave, photons, yerr=error)
        self.axesstd[13].errorbar(vegwave, vegflux, yerr=vegerr, color='red')
        self.axesstd[14].errorbar(wave, ftoabs, yerr=ftoabs_error, color='red')
        wave, flux, error, sname = self.calibrate_real(wave, photons, error, sname, wave,
                                                       ftoabs, ftoabs_error)  # real units spectra
        self.axesstd[11].errorbar(wave, flux, yerr=error)
        calib_standard = np.array((wave, flux, error))
        return ftoabs, calib_standard, ftoabs_error, sname, resid

    def calibrate_real(self, wave: np.ndarray,
                       photons: np.ndarray, error: np.ndarray,
                       name: str, wavereg: np.ndarray,
                       ftoabs: np.ndarray, ftoabserr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Calibrates the object spectra with the model flux conversion

        This method takes the wavelength and counts given and converts the counts to an absolute flux at 10 parsecs.
        The distance is taken from the listed table and scales the resulting flux by (distance / 10)^2 to place the
        object at 10pc.

        Parameters
        ----------
        wave : np.ndarray
            The 1D array of target wavelengths
        photons : np.ndarray
            The 1D array of target photon counts
        error : np.ndarray
            The 1D array of target photon count error
        name : str
            The name of the target
        wavereg : np.ndarray
            Region constraining the wavelength looked at
        ftoabs : np.ndarray
            The calibration function
        ftoabserr : np.ndarray
            The calibration function error
        """
        dist_std = {'g191b2b': 59.88, 'gd140': 15.81, 'gd153': 63.0, 'gd248': 35.5,
                    'g158100': 461.42, 'ross640': 15.89, 'hilt600': 2541.3}  # the distances to all stds observed
        name = self.get_targetname(name)  # get the shortname of the target (else keep name)
        tempname = name.split('_')[-1].lower().replace('-', '')  # if object is a standard convert it
        if tempname in dist_std.keys():
            name = tempname
        wave, photons, error = self.confining_region(wave, photons, error, wavereg)  # constrict regime
        flux = photons / ftoabs  # determine the absolute flux at the Earth
        if self.resolution == 'R0300R':  # to correct for second order contamination
            # correction = np.loadtxt('R300R_correction_function.txt', unpack=True, usecols=(1, ))
            correction = np.ones_like(flux)
            flux *= correction
        error = self.calibrate_errorprop(flux, error, ftoabserr, ftoabs)  # propagate the error on flux
        return wave, flux, error, name

    def hum_air(self) -> Tuple[float, float, float]:
        """Gets the humidity and airmass of the object

        When there are multiple observations there can be multiple humidity and airmass measurements, this method
        takes all the values and averages them
        """
        object_list = self.checkifspectra(glob.glob(self.ptoobj))  # list of objects
        hum = np.array([self.get_header_info(fname)[1] for fname in object_list])
        air = np.array([self.get_header_info(fname)[2] for fname in object_list])
        mjd = np.array([self.get_header_info(fname)[3] for fname in object_list])
        return round(np.mean(hum), 2), round(np.mean(air), 2), round(np.mean(mjd), 2)

    def humairdiff(self, line: str) -> Tuple[str, float]:
        """Finds the difference between airmass and humidity

        Compares the airmass and humidity of the observation with that of the standard

        Parameters
        ----------
        line : str
            Given a line in a file in which the name, humidity and airmass are stored
        """
        res = line.split('_')[2]  # resolution of the WD standard
        if res != self.resolution:
            return [line.strip('\n'), np.inf]  # don't use observations from different resolutions
        hum = float(line.split('_')[3])
        air = float(line.split('_')[4])  # humidity and airmass of WD standard
        mjd = float(line.split('_')[5])
        humdiff = abs(hum - self.humidity)  # difference between standard and target in humidity
        airdiff = abs(air - self.airmass)  # difference between standard and target in airmass
        mjddiff = abs(mjd - self.mjd)   # difference between standard and target in time
        return [line.strip('\n'), np.sqrt((humdiff / 60) ** 2 + (airdiff / 1.5) ** 2 + (mjddiff / 365) ** 2)]

    def closest_standard(self) -> str:
        """Finds the closest standard in terms of airmass and humidity

        Opens the listed file and determines the combined distance in airmass and humidity - space and determines
        which one has the least distance to the target.
        """
        with open('good_wds.txt', 'r+') as f:
            all_diffs = np.stack([self.humairdiff(line) for line in f])  # all standards checked
            lowest_diff_ind = np.argmin(all_diffs[:, 1])  # minimum difference index
        return all_diffs[lowest_diff_ind, 0]  # select the standard with the least difference

    def object(self, ptoobj: str) -> Tuple[np.ndarray, str, np.ndarray]:
        """Reduce the target spectra

        The target object, if there are multiple observations, is stacked into 3D and then unstacked on the median to
        the 2D CCD. It is then bias subtracted, flat fielded and bad pixel masked before being having its spectra
        extracted and background subtracted. Pixels are converted to wavelength via use of the corresponding arcmap.
        Photon count is converted to absolute flux at 10 pc by use of a calibration function derived from the standards
        to WD model relation and distance scaled.

        Parameters
        ----------
        ptoobj : str
            The UNIX path to the spectra being reduced
        """
        object_list = self.checkifspectra(glob.glob(ptoobj))  # list of objects
        tname = self.get_header_info(object_list[-1])[0]
        all_objects = np.stack([self.fopener(obj) for obj in object_list])
        median_object = self.med_stack(all_objects)  # median stack objects
        self.axesobj[0].imshow(median_object, cmap='cividis', origin='lower', aspect='auto')
        self.axesobj[0].set_title(f'Median Stack ({len(all_objects)} object/s)')
        all_objects = np.stack([self.bisub(self.master_bias, obj) for obj in all_objects])
        median_object = self.med_stack(all_objects)  # median stack bias subtracted objects
        self.axesobj[1].imshow(median_object, cmap='cividis', origin='lower', aspect='auto')
        flat_object = self.flat_field(self.master_flat, median_object)  # flat field the object
        self.axesobj[2].imshow(flat_object, cmap='cividis', origin='lower', aspect='auto')
        fixed_object = self.inf_fix(flat_object)  # fix infinite counts, acts as a bad pixel mask
        self.axesobj[3].imshow(fixed_object, cmap='cividis', origin='lower', aspect='auto')
        pixel, photons, resid, aptleft, aptcent, aptright = self.extract(fixed_object)  # extracts spectra
        self.axesobj[7].imshow(resid, cmap='coolwarm', origin='lower', aspect='auto',
                               extent=(self.coordinate - 25, self.coordinate + 25, self.pixlow, self.pixhigh))
        aptleftdiff = aptcent - aptleft
        aptrightdiff = aptright - aptcent
        aptcent = aptcent - 25 + self.coordinate
        aptleft = aptcent - aptleftdiff
        aptright = aptcent + aptrightdiff
        self.axesobj[7].plot(aptleft, pixel, color='black', lw=1, ls='--')
        self.axesobj[7].plot(aptright, pixel, color='black', lw=1, ls='--')
        self.axesobj[7].plot(aptcent, pixel, color='black', lw=1, ls='-')
        error = self.poisson(photons)  # creating the errors
        self.axesobj[8].errorbar(pixel, photons, yerr=error)
        wave, arcpix, arcwave = self.transform(pixel)  # transform pixels to wavelength
        self.axesobj[9].errorbar(wave, photons, yerr=error)
        self.axesobj[12].plot(arcpix, arcwave, color='red')
        wave, flux, error, tname = self.calibrate_real(wave, photons, error, tname,
                                                       self.master_standard[0],
                                                       self.ftoabs, self.ftoabs_error)  # real units spectra
        self.axesobj[10].errorbar(wave, flux, yerr=error)
        self.axesobj[13].errorbar(wave, self.ftoabs, yerr=self.ftoabs_error, color='red')
        calib_spectra = np.array((wave, flux, error))
        return calib_spectra, tname, resid

    def writing(self, ptostore: str) -> None:
        """Writes the standard used and object to files

        Parameters
        ----------
        ptostore : str
            The UNIX path to where the reduced spectra will be stored
        """
        tspec = Table(data=self.master_target.T)  # the actual target
        with open('alt_done.log', 'a+') as f:  # add to log file that the target has been reduced
            f.write(f'{self.ob}_{self.resolution}_{self.prog}\n')
        tspec.write(f'{ptostore}/objects/{self.ob}_{self.resolution}_{self.prog}_{self.target}.txt',
                    format='ascii.no_header', overwrite=True)
        tstand = Table(data=self.master_standard.T)  # the standard from the same observing block as standard
        tstand.write(f'{ptostore}/standards/'
                     f'{self.ob}_{self.resolution}_{self.prog}_{self.actual_standard}_{self.standard_name}.txt',
                     format='ascii.no_header', overwrite=True)
        tbias = Table(data=self.master_bias)
        tbias.write(f'{ptostore}/bias/'
                    f'{self.ob}_{self.resolution}_{self.prog}.txt',
                    format='ascii.no_header', overwrite=True)
        tflat = Table(data=self.master_flat)
        tflat.write(f'{ptostore}/flat/'
                    f'{self.ob}_{self.resolution}_{self.prog}.txt',
                    format='ascii.no_header', overwrite=True)
        tspec_resid = Table(data=self.target_residual)  # the cut out region around where the target spectra should be
        tspec_resid.write(f'{ptostore}/residuals/objects/{self.ob}_{self.resolution}_{self.prog}_{self.target}.txt',
                          format='ascii.no_header', overwrite=True)
        tstand_resid = Table(data=self.standard_residual)  # the cut out region around where the standard spectra is
        tstand_resid.write(f'{ptostore}/residuals/standards/'
                           f'{self.ob}_{self.resolution}_{self.prog}_{self.standard_name}.txt',
                           format='ascii.no_header', overwrite=True)
        tcalib = Table(data=(self.master_standard[0], self.ftoabs))  # the calibration function
        tcalib.write(f'{ptostore}/calib_funcs/'
                     f'{self.ob}_{self.resolution}_{self.standard_name}_{self.prog}_{self.target}.txt',
                     format='ascii.no_header', overwrite=True)
        return


class BPM:
    """Creates the bad pixel mask

    Bad pixel mask created using all the flats of a given resolution by comparing each value of each row of the CCD
    by its median, where if the difference is greater than the standard deviation, it is a bad pixel.
    This is then written to a respective file of BPM_(resolution)_python.txt as the full CCD.

    Methods
    -------
    row_stats : numpy.ndarray
    Finds the median and standard deviation of an array and returns an array of 1s and 0s, 0 where |val-median| > std
    make_bpm : str
    Creates the bad pixel mask using the given flats (for each resolution)
    """

    @staticmethod
    def row_stats(row: np.ndarray) -> np.ndarray:
        """Determines median and standard deviation from row. Returns bad pixel row

        Parameters
        ----------
        row : np.ndarray
            A row of the CCD
        """
        med = np.median(row)  # median of row
        std = np.std(row)  # standard deviation of row
        bpr = np.array([])
        for i in row:
            if abs(i - med) <= std:  # if the value differs from median by less than a standard deviation
                bpr = np.append(bpr, 1)  # it is a good pixel
            else:
                bpr = np.append(bpr, 0)
        return bpr

    def make_bpm(self, ptoallflats: str) -> None:
        """Makes the bad pixel mask if it does not already exist

        Parameters
        ----------
        ptoallflats : str
            The UNIX path to where all the flats for a given resolution are stored
        """
        res = ptoallflats.split('/')[1]
        flat_list = glob.glob(ptoallflats)  # list of flats
        all_flats = np.stack([OB.fopener(flat) for flat in flat_list])  # stack the CCDs in the 3rd dimension
        for flat in all_flats:
            print(np.min(flat), np.max(flat), np.median(flat), np.mean(flat), np.std(flat))
        median_flat = OB.med_stack(all_flats)  # unstack to 2D
        bpm = np.stack([self.row_stats(row) for row in median_flat])  # create the bad pixel mask row by row
        t = Table(data=bpm)
        t.write(f'BPM_{res}_python.txt', overwrite=True, format='ascii.no_header')
        return


def repeat_check() -> None:
    """Checks if an object has been observed in multiple observing blocks

    If an object has been observed multiple times, median stack their reduced spectra (if resolution is different)
    """

    def quick_error_prop(error_array: np.ndarray) -> np.ndarray:
        """Propagates error by adding in quadrature"""
        errs = np.stack([x ** 2 for x in error_array])
        return np.sqrt(np.sum(errs, axis=0))

    done_list = np.array(glob.glob('alt_redspec/objects/*txt'))
    restgt = np.array([])
    for f in done_list:
        f = f.strip('.txt').split('/')[-1].split('_')
        restgt = np.append(restgt, f[1] + '_' + f[-1])
    for i in range(len(restgt)):
        c, this = 0, np.array([], dtype=int)
        for j in range(len(restgt)):
            if restgt[i] == restgt[j]:
                c += 1
                this = np.append(this, j)
        if c > 1:
            name = str(restgt[i]).split('_')
            name = f'OB-comb_{name[0]}_GTC-comb_{name[1]}'
            try:
                all_wave = np.stack([np.loadtxt(done_list[n], unpack=True, usecols=(0,)) for n in this])
                all_flux = np.stack([np.loadtxt(done_list[n], unpack=True, usecols=(1,)) for n in this])
                all_error = np.stack([np.loadtxt(done_list[n], unpack=True, usecols=(1,)) for n in this])
                [os.system(f"rm {done_list[n]}") for n in this]
                t = Table(data=(np.median(all_wave, axis=0), np.median(all_flux, axis=0),
                                quick_error_prop(all_error)))
                t.write(f'alt_redspec/objects/{name}.txt', overwrite=True, format='ascii.no_header')
            except (FileNotFoundError, OSError):
                pass
    return


def main():
    """Main control module

    All classes are controlled from here. It checks required files are present and accounted for.

    Raises
    ------
    FileNotFoundError
        If a required file is not where it should be (normally current directory)
    """
    # TODO: Add more file checks and be more specfic than 'len', maybe another script

    # preamble #
    t0 = time.time()  # start a clock timer
    np.seterr(divide='ignore', invalid='ignore')  # numpy provides a warning for zero division, ignore this
    warnings.simplefilter("ignore", category=RuntimeWarning)  # another warning is for runtime, ignore this
    warnings.simplefilter('ignore', np.RankWarning)  # another warning about poorly fitting polynomial, ignore
    ob_list = glob.glob('Raw/R**/G**/OB*')  # list of observing blocks

    # check requirements #
    if glob.glob('*arcmap.txt').__len__() != 2:
        raise FileNotFoundError('Cannot find pixel - wave conversion in environment')

    # check BPM and make otherwise #
    if glob.glob('BPM_*_python.txt').__len__() != 2:
        print('No bad pixel masks present, creating now from all flats per resolution.')
        bpm = BPM()
        bpm.make_bpm('Raw/R2500I/G**/OB**/flat/0*fits')  # make the R2500I BPM
        bpm.make_bpm('Raw/R0300R/G**/OB**/flat/0*fits')  # make the R300R BPM
        print('Made bad pixel masks.')

    # prompt user if they want to repeat all reductions #
    do_all = args.do_all
    if do_all:
        if len(glob.glob('alt_redspec/**/*txt')) > 0:
            os.system("rm alt_redspec/**/*txt")  # delete the spectra
        f = open('alt_done.log', 'w+')  # empty the log file
        f.close()

    # checking which files have already been reduced #
    ob_list = np.array(ob_list)
    done_list = np.array([], dtype=bool)
    for obs in ob_list:
        with open('alt_done.log', 'r') as f:
            for line in f:
                if obs.split('/')[-1] in line and obs.split('/')[1] in line and obs.split('/')[2] in line:
                    done_list = np.append(done_list, False)
                    break
            else:
                done_list = np.append(done_list, True)
    ob_list = ob_list[done_list]
    ob_list = [ob_list[0], ]

    # thread the unreduced files #
    if len(ob_list):
        avail_cores = multiprocessing.cpu_count() - 1 or 1  # available cores to thread over
        if len(ob_list) < avail_cores:
            avail_cores = len(ob_list)
        print(f'Threading over {avail_cores} cores.')
        pool = multiprocessing.Pool(processes=avail_cores)
        pool.map(OB, ob_list)
        pool.close()
        # repeat_check()
        print('Done with spectra.')
        print(f'Run took {round((time.time() - t0) / 60, 1)} minutes.')
    else:
        print(f'Process took {round(time.time() - t0, 1)} seconds.')

    # prompt user if they want to make all the plots
    do_plot = args.do_plots
    if do_plot:
        sys.dont_write_bytecode = True
        tplot0 = time.time()
        import alt_plot
        alt_plot.main()
        print(f'Done with plotting in {round((time.time() - tplot0) / 60, 1)} minutes.')
    print(f'Total time taken was {round((time.time() - t0) / 60, 1)} minutes.')
    return


if __name__ == '__main__':  # if called as script, run main module
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    myargs.add_argument('-a', '--do-all', action='store_true', default=False, help='Do all spectra?')
    myargs.add_argument('-p', '--do-plots', action='store_true', default=False, help='Generate Further Plots?')
    args = myargs.parse_args()
    main()
