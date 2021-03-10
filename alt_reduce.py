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
    * .config  -- file containing config arguments
    * BPM_(resolution)_python.txt       -- A file for the bad pixel masks for the resolution (else on

Classes
-------
OB : str
    The full class, passed the string to the observing block and hence reduces the spectra in that observing block
BPM : str
    Determines the bad pixel mask if the corresponding resolution mask does not exist in the current environment

Methods
-------
main
    Main function of the script to make bad pixel mask and reduce the spectra
"""
# TODO: Update some of the documentation
import numpy as np  # general mathematics and array handling
from numpy.polynomial.polynomial import Polynomial as Poly
from astropy.io.fits import getdata, getheader
from astropy.table import Table  # opening files as data tables
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.interpolate import UnivariateSpline as Spline3
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import pandas as pd
from tqdm import tqdm

import argparse
import glob  # equivalent to linux 'ls'
import warnings  # used for suppressing an annoying warning message about runtime
import multiprocessing  # used to overcome the inherent python GIL
import time  # timing processing
from typing import Tuple, Sequence  # used for type hinting
import os
import json


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
    checkifspectra(spectral_list)
        Ascertains that the spectra fits file is not an acquisition (they are in the same directory)
    flat_field(data, flat)
        Divides the spectral CCD by the flat field
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
        self.logger(f'Resolution {self.resolution}\nProgramme {self.prog}\nObserving block {self.ob}', w=True)
        if self.target_check():
            # initialising
            pbar = tqdm(total=100, desc=f'{self.resolution}/{self.prog}/{self.ob}')
            self.figobj, self.axesobj = plt.subplots(4, 4, figsize=(16, 10), dpi=300)
            self.axesobj = self.axesobj.flatten()
            self.figstd, self.axesstd = plt.subplots(4, 4, figsize=(16, 10), dpi=300)
            self.figobjarc, self.axesobjarc = plt.subplots(figsize=(8, 5), dpi=300)
            self.figstdarc, self.axesstdarc = plt.subplots(figsize=(8, 5), dpi=300)
            self.axesstd = self.axesstd.flatten()
            # pixel limits
            self.pixlow, self.pixhigh = self.pixel_constraints()
            pbar.update(5)
            self.logger(f'Use pixels from {self.pixlow} to {self.pixhigh}')
            # bias
            self.ptobias = ptodata + '/bias/0*fits'  # path to biases
            self.logger(f'There are {len(glob.glob(self.ptobias))} bias files')
            self.master_bias = self.bias(self.ptobias)  # creates the master bias file
            pbar.update(5)
            biasplot = self.axesobj[4].imshow(self.master_bias, cmap='BuPu', origin='lower', aspect='auto')
            plt.colorbar(biasplot, ax=self.axesobj[4])
            # flat
            self.ptoflats = ptodata + '/flat/0*fits'  # path to flats
            self.logger(f'There are {len(glob.glob(self.ptoflats))} flats files')
            self.master_flat = self.flat(self.ptoflats, self.master_bias)  # creates the master flat file
            pbar.update(5)
            flatplot = self.axesobj[5].imshow(self.master_flat, cmap='BuPu', origin='lower', aspect='auto')
            plt.colorbar(flatplot, ax=self.axesobj[5])
            # bad pixel mask
            self.bpm = self.bpm_applying(np.ones_like(self.master_flat))
            pbar.update(5)
            bpmplot = self.axesobj[6].imshow(self.bpm, cmap=ccd_bincmap, origin='lower', aspect='auto')
            plt.colorbar(bpmplot, ax=self.axesobj[6])
            # header info and further initialisation
            self.ptoobj = ptodata + '/object/0*fits'  # path to object
            self.logger(f'There are {len(glob.glob(self.ptoobj))} object files')
            self.humidity, self.airmass, self.mjd = self.hum_air()  # average humidity and airmass of object obs
            pbar.update(5)
            self.logger(f'Humidity {self.humidity}\nAirmass {self.airmass}\nMJD {self.mjd}')
            self.ptostds = ptodata + '/stds/0*scopy.fits'  # path to standard
            self.logger(f'There are {len(glob.glob(self.ptostds))} standards files')
            self.ptoarcs = ptodata + '/arc/0*fits'  # path to arcs
            self.logger(f'There are {len(glob.glob(self.ptoarcs))} arcs files')
            # standard
            self.logger('The standard is being reduced and analysed:')
            self.haveaperture = False
            self.ftoabs, self.master_standard, self.ftoabs_error, self.standard_name, \
            self.standard_residual, self.cpix, self.aptleft, self.aptright = self.standard()  # standard reduction
            self.haveaperture = True
            pbar.update(35)
            self.logger(f'Name of standard being used was {self.standard_name.upper()}')
            # object
            self.logger('The object is now being reduced:')
            self.master_target, self.target, self.target_residual = self.object()  # reduces target
            pbar.update(30)
            # writing files and creating plots
            self.logger(f'The target was {self.target}')
            self.fig_formatter()  # formats the plots (titles, labels)
            self.writing('alt_redspec')  # writes reduced spectra to files
            pbar.update(5)
            pbar.update(5)
            pbar.close()
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
        try:
            tinfo = Table.read('Master_info_correct_cm.csv')  # table containing observation information
        except (OSError, FileNotFoundError):
            return True
        else:
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
        sname = sname.strip()  # remove whitespace
        try:
            tinfo = Table.read('Master_info_correct_cm.csv')  # table containing header names and shortnames
        except (OSError, FileNotFoundError):
            pass
        else:
            for row in tinfo:
                if row['Target'].strip() == sname:  # if the header names match
                    sname = row['SHORTNAME']  # take the shortname as the target name
                    break
        return sname

    @staticmethod
    def get_pixcoord(data: np.ndarray) -> int:
        """Gets the pixel coordinate

        The row pixel where the spectra is should be recorded in the listed file (3 columns, programme ID,
        observation block, and pixel coordinate.
        """
        coord = np.nanmedian(np.argmax(data, axis=-1)).astype(int)
        return coord  # pixel to index conversion

    @staticmethod
    def pixel_constraints() -> Tuple[int, int]:
        """Finds the constraints for the respective resolution

        Using the resolution of this object, find the limits on the dispersion axis at which to extract spectra
        """
        xmin, xmax = config.minpix, config.maxpix
        return xmin, xmax

    @staticmethod
    def bisub(data: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Bias subtract from each ccd given

        Elementwise subtraction of the entire CCD, pixel by pixel

        Parameters
        ----------
        bias : np.ndarray
            The bias from same observing block as the object
        data : np.ndarray
            The full CCD that will have the bias subtracted
        """
        return np.subtract(data, bias)

    def bpm_applying(self, data: np.ndarray) -> np.ndarray:
        """Applies the bad pixel mask

        Elementwise multiplication where 0 is a bad pixel and 1 (unchanged) for good pixels

        Parameters
        ----------
        data : np.ndarray
            The full CCD that needs to be masked
        """
        mask = np.loadtxt(f'BPM_{self.resolution}_python.txt')
        return np.multiply(mask, data)  # BPM is array of 1=good, 0=bad

    @staticmethod
    def fopener(fname: str) -> np.ndarray:
        """Opens the fits file using astropy

        Parameters
        ----------
        fname : str
            The relative or full string to the file to be opened
        """
        data = getdata(fname, ext=2)
        return data

    def fopenbisub(self, fname: str, bias: np.ndarray) -> np.ndarray:
        """
        Uses the file opening method and bias subtracting method as all-in-one convenience function

        Parameters
        ----------
        bias: np.ndarray
            The bias numpy array
        fname : str
            The relative or full string to the file to be opened
        """
        unsubbed_arr = self.fopener(fname)
        return self.bisub(unsubbed_arr, bias)  # opens and then bias subtracts the file using the listed methods

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
        low = config.cpix - config.stripewidth - 1
        high = config.cpix + config.stripewidth - 1
        if low < 0:
            low = 0
        if high > data.shape[1]:
            high = data.shape[1]
        dtrimmed = data[self.pixlow: self.pixhigh, low: high]
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
        all_flats = np.stack([self.fopenbisub(flat, bias) for flat in flat_list])
        median_flat = self.med_stack(all_flats)  # determines bias subtracted median flat
        bpm_flat = self.bpm_applying(median_flat)  # apply bad pixel mask
        return self.normalise(bpm_flat)  # normalised flat

    @staticmethod
    def identify(pixel: np.ndarray, dataline: np.ndarray, lamp: str) -> pd.DataFrame:
        """
        Identifies the lines in an arc and compares to line list

        Parameters
        ----------
        pixel: np.ndarray
            The 1D array of spectra pixels
        dataline: np.ndarray
            The 1D array of the arc to have lines identified in
        lamp: str
            The object keyword in the header for which lamp is being checked

        Returns
        -------
        pix_wave: pd.DataFrame
            A dataframe of columns pixel to wavelength
        """
        linelist = pd.read_csv(f'lablinelist/{lamp}I.csv')
        linelist = linelist[np.logical_and(linelist.wave > config.minwave, linelist.wave < config.maxwave)].copy()
        linelist.sort_values('intens', ascending=False, inplace=True, ignore_index=True)  # sort down by intensity
        xhigh = np.linspace(np.min(pixel), np.max(pixel), pixel.size * 100)
        sp = Spline3(pixel, dataline)
        spfit = sp(xhigh)
        cpeak, wherepk, increasing = 0, np.empty(0), True
        comp = np.mean(spfit) + np.std(spfit)
        pntcomp = np.mean(spfit)
        for i, pnt in enumerate(spfit):
            i = xhigh[i]
            if pnt > pntcomp:
                increasing = True
            else:
                if pntcomp > comp and increasing:
                    if not len(wherepk) or i - wherepk[-1] > 3:  # ignore blended
                        wherepk = np.append(wherepk, i)
                        cpeak += 1
                increasing = False
            pntcomp = pnt

        linewaves = np.empty(0)
        for i, row in linelist.iterrows():
            if len(linewaves) == cpeak:
                break
            if not linewaves.size or np.all(np.abs(np.subtract(linewaves, row.wave)) > 5):  # ignore blended
                linewaves = np.append(linewaves, row.wave)

        pix_wave = pd.DataFrame({'pixel': wherepk, 'wave': np.sort(linewaves)})
        return pix_wave

    def solution_fitter(self, df: pd.DataFrame, ax: plt.Axes) -> Poly:
        """
        Iteratively fits to arc solution

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe of pixel to wavelength
        ax : plt.Axes
            The plot axes to put the residuals on

        Returns
        -------
        pcomb: Poly
            The combined polynomial solution
        """
        linfit: Poly = Poly.fit(df.pixel.values, df.wave.values, deg=1)  # linear fit
        residual = linfit(df.pixel.values) - df.wave.values
        df = df[np.logical_and(np.abs(residual) < 100,
                               np.logical_and(df.pixel > self.pixlow + 100,
                                              df.pixel < self.pixhigh - 100))].copy()
        linfit = Poly.fit(df.pixel.values, df.wave.values, deg=1)  # linear fit
        residual = linfit(df.pixel.values) - df.wave.values
        thirdorder: Poly = Poly.fit(df.pixel.values, residual, deg=3)
        ax.plot(df.pixel, thirdorder(df.pixel.values), 'yx', label='Linear Residual 1')
        ax.plot(*thirdorder.linspace(), 'y--', label='3rd Order Fit 1')
        pcomb = linfit - thirdorder
        residual = pcomb(df.pixel.values) - df.wave
        ax.plot(df.pixel, residual, 'mx', label='Combined Residual 1')
        iqr = np.subtract(*np.quantile(residual, [.75, .25]))
        ax.fill_between(df.pixel.values, -2*iqr, 2*iqr, color='cyan', label='2X IQR 1', alpha=0.5)
        df = df[np.logical_and(np.abs(residual) < 2 * iqr,
                               np.logical_and(df.pixel > self.pixlow + 100,
                                              df.pixel < self.pixhigh - 100))].copy()
        residual = residual[np.abs(residual) < 2 * iqr]
        weights = np.divide(1, np.power(residual, 3), where=residual != 0)
        # do again from start, now with weights and some bad data removed
        linfit = Poly.fit(df.pixel.values, df.wave.values, deg=1, w=weights)  # linear fit
        residual = linfit(df.pixel.values) - df.wave.values
        thirdorder = Poly.fit(df.pixel.values, residual, deg=3, w=weights)
        ax.plot(df.pixel, thirdorder(df.pixel.values), 'rx', label='Linear Residual 2')
        ax.plot(*thirdorder.linspace(), 'r--', label='3rd Order Fit 2')
        pcomb = linfit - thirdorder
        residual = pcomb(df.pixel.values) - df.wave
        rmsd = np.sqrt(np.sum(np.square(residual)) / residual.size)
        iqr = np.subtract(*np.quantile(residual, [.75, .25]))
        ax.fill_between(df.pixel.values, -2 * iqr, 2 * iqr, color='blue', label='2X IQR 2', alpha=0.25)
        rmsdiqr = rmsd / iqr
        self.logger(f'Arc solution RMSDIQR {rmsdiqr}')
        ax.plot(df.pixel, residual, 'bx', label='Combined Residual 2')
        return pcomb

    def arc_solution(self, pixel: np.ndarray, cpix: np.ndarray, ptoarcs: str) -> np.ndarray:
        """
        Generates the wavelength solution from the arcs

        Parameters
        ----------
        pixel: np.ndarray
            The array of pixels corresponding to the spectra
        cpix: np.ndarray
            The pixel where the centre of the extraction aperture lies
        ptoarcs: str
            The path to the directory holding the arc files

        Returns
        -------
        wave: np.ndarray
            The wavelengths converted from pixels via the wavelength solution
        """
        if self.haveaperture:
            ax = self.axesobj[12]
            ax2 = self.axesobjarc
        else:
            ax = self.axesstd[12]
            ax2 = self.axesstdarc
        arcfiles = glob.glob(ptoarcs)
        all_pix_wave = pd.DataFrame(columns=('pixel', 'wave'))
        for i, arc in enumerate(arcfiles):
            arcdata = getdata(arc, ext=2)  # extract whole arc
            arcdata = np.subtract(arcdata, self.master_bias)  # bias subtract
            arcdata = np.divide(arcdata, self.master_flat, where=self.master_flat != 0)  # flat field
            arcdata = arcdata[self.pixlow: self.pixhigh]
            arccut = arcdata[np.arange(len(arcdata)), cpix.astype(int)].flatten()  # just extraction pixel
            lamp = getheader(arc, ext=0)['OBJECT'].split('_')[-1].lower()
            pix_wave = self.identify(pixel, arccut, lamp)
            all_pix_wave = all_pix_wave.append(pix_wave).reset_index(drop=True)
        all_pix_wave.sort_values('pixel', inplace=True, ignore_index=True)
        soln = self.solution_fitter(all_pix_wave, ax2)
        wave = soln(pixel)
        ax.plot(pixel, wave, 'k-')
        ax.plot(all_pix_wave.pixel, all_pix_wave.wave, 'kx')
        return wave

    @staticmethod
    def gaussian(_x: np.ndarray, _amp: float, cen: float, wid: float):
        """
        Gaussian equation

        Parameters
        ----------
        _x: np.ndarray
            The x values
        _amp : float
            Amplitude
        cen: float
            Mean value
        wid: float
            Standard deviation

        Returns
        -------
        _: np.ndarray
            Gaussian value for given x points
        """
        return _amp * np.exp(-(_x - cen) ** 2 / wid)

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
            head = getheader(spectra, ext=0)
            if head['GRISM'] == 'OPEN':  # i.e. acquisiton
                bo_arr = np.append(bo_arr, False)
            else:
                bo_arr = np.append(bo_arr, True)
        return spectral_list[bo_arr]  # only use files that are actual spectra

    @staticmethod
    def flat_field(data: np.ndarray, flat: np.ndarray) -> np.ndarray:
        """
        Apply the normalised flat field and act as bad pixel mask where the flat is 0

        Parameters
        ----------
        flat : np.ndarray
            The full CCD of the flat to be used
        data : np.ndarray
            The full CCD that will needs to be flat fielded
        """
        return np.divide(data, flat, where=flat != 0)  # returns 0 on the bad pixel masks

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
        lbound = cpix - 5 if cpix - 5 >= 0 else 0
        rbound = cpix + 5 if cpix + 5 <= len(segment) else len(segment)
        cpix = np.argmax(segment[lbound:rbound]) + lbound  # the new central row index where the peak is in 10
        return cpix

    @staticmethod
    def find_back(segment: np.ndarray) -> float:
        """
        Finds the background value using an iterative mode

        Parameters
        ----------
        segment: np.ndarray
            Aperture axis cut off

        Returns
        -------
        backmode: float
            The modal background count representing the sky
        """
        backmode = np.nanmedian(segment)  # start with median value as background
        i = len(segment)  # start bin size as segment length
        significant = False
        while not significant:
            hist, edges = np.histogram(segment, bins=i)  # bin the values
            significant = np.max(hist) > np.mean(hist) + np.std(hist) * 2  # if the mode is 2 sigma above the mean
            if i == 5:  # if the number of bins = number of pixels
                significant = True
            if significant:  # if either of last two true
                modeind = np.argmax(hist)  # which index has the modal value
                backmode = np.mean(edges[modeind: modeind + 2])  # the mean value of that bin is the background
            else:
                i -= 5  # decrease bin size by 5 if the mode is not yet significant
        return backmode

    def peak_average(self, segment: np.ndarray, cpix: int, ind: int, jdict: dict) -> Tuple[float, int, int, int, dict]:
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
        ind : int
            Index along dispersion axis
        jdict : dict
            Dictionary of extraction results
        """
        jdict[ind] = thisdict = {}
        cpix = round(cpix)
        if np.array(segment == np.zeros_like(segment)).all():  # bad pixel rows
            return 0, 25, 25, 25
        backmode = self.find_back(segment)
        if self.haveaperture:
            leftwidth, rightwidth = self.aptleft[ind], self.aptright[ind]
        else:
            leftwidth, rightwidth = 10, 10
        minind = cpix - leftwidth if cpix - leftwidth >= 0 else 0
        maxind = cpix + rightwidth if cpix + rightwidth <= len(segment) else len(segment)
        backsub = np.array([self.back_subtract(i, backmode) for i in segment])  # subtract the background from row
        minreg = np.min(np.abs(backsub))
        region = np.abs(np.subtract(backsub, minreg)).astype(float)
        xsmall = np.arange(len(region), dtype=float)
        xbig = np.linspace(0, len(region), 1000, endpoint=False, dtype=float)
        amp = np.nanmax(region[minind: maxind]).astype(float)
        thisdict['xsmall'] = xsmall.tolist()
        thisdict['xbig'] = xbig.tolist()
        yvals = np.interp(xbig, xsmall, region)
        thisdict['region'] = segment.astype(float).tolist()
        thisdict['ybig'] = (yvals + backmode + minreg).astype(float).tolist()
        p0 = [amp, cpix, 2]
        try:
            gfit = curve_fit(self.gaussian, xbig, yvals, p0=p0,
                             bounds=([0, minind, 0.5], [1.2 * amp, maxind, 3]))[0]
        except ValueError:
            gfit = p0
        except RuntimeError:
            gfit = p0
        yvals = self.gaussian(xbig, *gfit)
        xstd = gfit[2]
        hwhm = np.sqrt(2 * np.log(2)) * xstd
        bigcpix = cpix = gfit[1]
        bigcpix_mhwhm = bigcpix - 0.75 * hwhm
        bigcpix_phwhm = bigcpix + 0.75 * hwhm
        try:
            cpix_mhwhm = np.flatnonzero(xsmall > bigcpix_mhwhm)[0]
        except IndexError:
            cpix_mhwhm = 0
        try:
            cpix_phwhm = np.flatnonzero(xsmall > bigcpix_phwhm)[0]
        except IndexError:
            cpix_phwhm = len(xsmall)
        try:
            bpix_lowind = np.flatnonzero(xbig > bigcpix_mhwhm)[0]
        except IndexError:
            bpix_lowind = 0
        try:
            bpix_highind = np.flatnonzero(xbig > bigcpix_phwhm)[0]
        except IndexError:
            bpix_highind = len(xbig)
        thisdict['yfit'] = (yvals + backmode + minreg).astype(float).tolist()
        thisdict['params'] = [float(bigcpix_mhwhm), float(bigcpix), float(bigcpix_phwhm),
                              float(amp), float(minreg), float(backmode)]
        signal = np.trapz(yvals[bpix_lowind: bpix_highind] + minreg)
        return signal, cpix_mhwhm, cpix, cpix_phwhm, backmode, jdict

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
        backmode = self.find_back(segment)
        backsub = np.array([self.back_subtract(i, backmode) for i in segment])
        cpix = cpix + 1 if cpix == 0 else cpix
        cpix = cpix - 1 if cpix == len(segment) else cpix
        lbound = cpix - 5 if cpix - 5 >= 0 else 0
        rbound = cpix + 5 if cpix + 5 <= len(segment) else len(segment)
        try:
            relevel = 0.1 * np.max(backsub[lbound:rbound])  # 10 percent of the peak signal
        except ValueError:
            relevel = backsub[cpix]
        try:
            minind = np.flatnonzero(backsub[lbound:cpix] < relevel)[-1] + lbound  # lower aperture limit
            maxind = np.flatnonzero(backsub[cpix:rbound] < relevel)[0] + cpix  # upper aperture limit
        except IndexError:
            minind = lbound
            maxind = rbound
        leftwidth = cpix - minind if cpix - minind > 1 else 1
        rightwidth = maxind - cpix if maxind - cpix > 1 else 1
        return leftwidth, rightwidth

    def extract(self, data: np.ndarray, jdict: dict, coord: int) -> Tuple[np.ndarray, np.ndarray,
                                                                          np.ndarray, np.ndarray,
                                                                          np.ndarray, np.ndarray,
                                                                          np.ndarray, dict]:
        """Extracts the spectrum

        Take a 50 pixel slice around the central pixel with a sub-section of rows selected from the CCD based on the
        arc map (e.g. to ensure only first order diffraction in R300R). Then extract and background subtract the
        signal from that slice on a row by row basis.

        Parameters
        ----------
        data : np.ndarray
            The full CCD of the observation, to be sliced and extracted from
        jdict : dict
            Dictionary of extraction results
        coord: int
            Central pixel
        """
        data = data[self.pixlow:self.pixhigh,
                    coord - config.stripewidth // 2: coord + 1 + config.stripewidth // 2]  # slicing spectra
        pixels = np.arange(self.pixlow, self.pixhigh) + 1
        peaks, aptleft, aptcent, aptright, background = np.empty_like(pixels), np.empty_like(pixels), \
            np.empty_like(pixels), np.empty_like(pixels), np.empty_like(pixels)
        for i, row in enumerate(data):
            if not i:
                cpix = len(row) // 2
            else:
                cpix = aptcent[i - 1]
            peak_extract = self.peak_average(row, cpix, i, jdict)
            peaks[i] = peak_extract[0]
            aptleft[i] = peak_extract[1]
            aptcent[i] = peak_extract[2]
            aptright[i] = peak_extract[3]
            background[i] = peak_extract[4]
            jdict = peak_extract[5]
        return pixels, peaks, data, aptleft, aptcent, aptright, background, jdict

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

    def model_data(self, whichstd: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the data from the relevant model

        Parameters
        ----------
        whichstd: str
            Name of the standard

        Returns
        -------
        modwave: np.ndarray
            The model wavelength
        modflux: np.ndarray
            The model flux
        moderr: np.ndarray
            The model flux errors
        """
        modwave, modflux = np.loadtxt(f'calib_models/{whichstd}_mod.txt', unpack=True, usecols=(0, 1))  # load model
        # wave, f_lambda, f_nu are the columns
        self.logger(f'Final wavelength of model is {modwave.max()}A')
        moderr = np.ones_like(modwave) * (np.std(modflux) / len(modflux))  # determine the error
        return modwave, modflux, moderr

    def vector_func(self, whichstd: str, wave: np.ndarray,
                    flux: np.ndarray, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divides the standard by a model to get a vector constant

        Loads the model that calibrates with the best standard for these objects, interpolates them down to
        the resolution of the standard spectra. Then finds the calibration function and propagates the error
        into that (error determined from the models).

        Parameters
        ----------
        whichstd : str
            A string which is the name of the that has standard observations and a calibration model
        wave : np.ndarray
            A 1D array of wavelengths of the standard
        flux : np.ndarray
            A 1D array of photon counts of the standard
        error : np.ndarray
            A 1D array of errors on the standard counts
        """
        modwave, modflux, moderr = self.model_data(whichstd)
        modflux = np.interp(wave, modwave, modflux)
        moderr = np.interp(wave, modwave, moderr)
        modwave = np.interp(wave, modwave, modwave)
        ftoabs = flux / modflux  # the calibration function as counts to absolute flux
        comb_error = self.calibrate_errorprop(ftoabs, error, moderr, modflux)  # propagate the error into function
        return wave, flux, error, ftoabs, comb_error, modwave, modflux, moderr

    @staticmethod
    def get_header_info(fname: str) -> Tuple[str, float, float, float]:
        """Gets the standard file name

        Parameters
        ----------
        fname : str
            The UNIX path to the file being opened as a string
        """
        head = getheader(fname, ext=0)  # the observational information on OSIRIS is on the first HDU
        return head['OBJECT'].rstrip(), head['HUMIDITY'], head['AIRMASS'], head['MJD-OBS']

    def json_handler(self, objname: str, perm: str, jobj: dict = None):
        """
        Handles the creation and editing of json files (used for live plots)

        Parameters
        ----------
        objname : str
            Name of the object being handled
        perm : str
            Permissions to open json with
        jobj : dict
            Object to dump into json if writing

        """
        if perm not in ('w', 'r'):
            raise ValueError(f'Unsure on permission "{perm}"')
        with open(f'alt_redspec/jsons/{self.ob}_{self.resolution}_{self.prog}_{objname}.json', perm) as jfile:
            if perm == 'w':
                json.dump(jobj, jfile)
                return
            elif perm == 'r':
                jobj: dict = json.load(jfile)
                return jobj

    def standard(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduces standard and creates conversion to absolute units

        The observation is stacked on its median, bias subtracted, bad pixel masked and flat fielded
        before the signal is extracted along
        the dispersion axis with pixels converted to wavelength. Using the model, a calibration function of counts
        to flux is determined.
        """
        # initialise
        self.logger(f'The standard has {len(glob.glob(self.ptostds))} standard files')
        biasplot = self.axesstd[4].imshow(self.master_bias, cmap='BuPu', origin='lower', aspect='auto')
        plt.colorbar(biasplot, ax=self.axesstd[4])
        flatplot = self.axesstd[5].imshow(self.master_flat, cmap='BuPu', origin='lower', aspect='auto')
        plt.colorbar(flatplot, ax=self.axesstd[5])
        bpmplot = self.axesstd[6].imshow(self.bpm, cmap=ccd_bincmap, origin='lower', aspect='auto')
        plt.colorbar(bpmplot, ax=self.axesstd[6])
        standard_list = self.checkifspectra(glob.glob(self.ptostds))  # list of standards)
        sname = self.get_header_info(standard_list[-1])[0]  # gets name of standard
        sname = sname.split('_')[-1].lower().replace('-', '')  # converting standard name from header to model name
        # stack raw data
        all_standards = np.stack([self.fopener(obj) for obj in standard_list])
        median_standard = self.med_stack(all_standards)  # median stack objects
        self.axesstd[0].imshow(median_standard, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto')
        self.axesstd[0].set_title(f'Median Stack ({len(all_standards)} standard/s)')
        # bias subtract
        all_standards = np.stack([self.bisub(std, self.master_bias) for std in all_standards])
        median_standard = self.med_stack(all_standards)  # median stack the bias subtracted standards
        stdcoord = self.get_pixcoord(median_standard)  # centre pixel
        self.logger(f'Central pixel for standard around {stdcoord + 1}')
        self.axesstd[1].imshow(median_standard, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto')
        # flat field
        flat_standard = self.flat_field(median_standard, self.master_flat)  # flat field the standard
        self.axesstd[2].imshow(flat_standard, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto')
        # apply bad pixel mask
        fixed_standard = flat_standard
        self.axesstd[3].imshow(fixed_standard, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto')
        # extract spectra
        self.json_handler(sname, 'w', {})
        jdict = self.json_handler(sname, 'r')
        pixel, photons, resid, aptleft, aptcent, aptright, back, jdict = self.extract(fixed_standard, jdict, stdcoord)
        if dojsons:
            self.json_handler(sname, 'w', jdict)
        del jdict
        outcpix = aptcent
        aptleftdiff = aptcent - aptleft
        aptrightdiff = aptright - aptcent
        aptcent = aptcent - config.stripewidth // 2 + stdcoord + 1
        aptleft = aptcent - aptleftdiff
        aptright = aptcent + aptrightdiff
        reducplot = self.axesstd[7].imshow(resid, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                                           extent=(stdcoord - config.stripewidth // 4,
                                                   stdcoord + 1 + config.stripewidth // 4,
                                                   self.pixlow, self.pixhigh))
        plt.colorbar(reducplot, ax=self.axesstd[7])
        self.axesstd[7].plot(aptleft, pixel, color='black', lw=1, ls='--')
        self.axesstd[7].plot(aptright, pixel, color='black', lw=1, ls='--')
        self.axesstd[7].plot(aptcent, pixel, color='black', lw=1, ls='-')
        error = self.poisson(photons)  # creating the errors
        self.axesstd[8].errorbar(pixel, photons + back, yerr=error, color='green')
        self.axesstd[8].plot(pixel, back, color='orange')
        # wavelength calibrate
        wave = self.arc_solution(pixel, aptcent, self.ptoarcs)
        self.axesstd[9].errorbar(wave, photons, yerr=error)
        # create flux calibration function
        wave, photons, error, ftoabs, ftoabs_error,\
        modwave, modflux, moderr = self.vector_func(sname, wave, photons, error)  # calib function
        self.axesstd[10].errorbar(wave, photons, yerr=error)
        self.axesstd[13].errorbar(modwave, modflux, yerr=moderr, color='red')
        self.axesstd[14].errorbar(wave, ftoabs, yerr=ftoabs_error, color='red')
        # calibrate standard fluxes
        wave, flux, error, sname = self.calibrate_real(wave, photons, error, sname,
                                                       ftoabs, ftoabs_error)  # real units spectra
        self.axesstd[11].errorbar(wave, flux, yerr=error)
        calib_standard = np.array((wave, flux, error))
        return ftoabs, calib_standard, ftoabs_error, sname, resid,\
            outcpix, aptleftdiff, aptrightdiff

    def calibrate_real(self, wave: np.ndarray,
                       photons: np.ndarray, error: np.ndarray,
                       name: str,
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
        flux = photons / ftoabs  # determine the absolute flux at the Earth
        error = self.calibrate_errorprop(flux, error, ftoabserr, ftoabs)  # propagate the error on flux
        return wave, flux, error, name

    def hum_air(self) -> Tuple[float, float, float]:
        """Gets the humidity and airmass of the object

        When there are multiple observations there can be multiple humidity and airmass measurements, this method
        takes all the values and averages them
        """
        object_list = self.checkifspectra(glob.glob(self.ptoobj))  # list of objects
        hum, air, mjd = np.empty(0), np.empty(0), np.empty(0)
        for fname in object_list:
            _hum, _air, _mjd = self.get_header_info(fname)[1:]
            _params = _hum, _air, _mjd
            params = [hum, air, mjd]
            for i, param in enumerate(params):
                param = np.append(param, _params[i])
                params[i] = param
            hum, air, mjd = params
        return round(np.mean(hum), 2), round(np.mean(air), 2), round(np.mean(mjd), 2)

    def object(self) -> Tuple[np.ndarray, str, np.ndarray]:
        """Reduce the target spectra

        The target object, if there are multiple observations, is stacked into 3D and then unstacked on the median to
        the 2D CCD. It is then bias subtracted, flat fielded and bad pixel masked before being having its spectra
        extracted and background subtracted. Pixels are converted to wavelength via use of the corresponding arcmap.
        Photon count is converted to absolute flux at 10 pc by use of a calibration function derived from the standards
        to model relation and distance scaled.
        """
        # initialise
        object_list = self.checkifspectra(glob.glob(self.ptoobj))  # list of objects
        tname = self.get_header_info(object_list[-1])[0]
        # raw data
        all_objects = np.stack([self.fopener(obj) for obj in object_list])
        median_object = self.med_stack(all_objects)  # median stack objects
        self.axesobj[0].imshow(median_object, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto')
        self.axesobj[0].set_title(f'Median Stack ({len(all_objects)} object/s)')
        # centre coordinate
        coordinate = self.get_pixcoord(median_object)  # find pixel coordinate
        self.logger(f'Central pixel for object around {coordinate + 1}')
        # bias subtract
        all_objects = np.stack([self.bisub(obj, self.master_bias) for obj in all_objects])
        median_object = self.med_stack(all_objects)  # median stack bias subtracted objects
        self.axesobj[1].imshow(median_object, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto')
        # flat field
        flat_object = self.flat_field(median_object, self.master_flat)  # flat field the object
        self.axesobj[2].imshow(flat_object, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto')
        # apply bad pixel mask
        fixed_object = flat_object
        self.axesobj[3].imshow(fixed_object, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto')
        # extract spectra
        self.json_handler(tname, 'w', {})
        jdict = self.json_handler(tname, 'r')
        pixel, photons, resid,\
        aptleft, aptcent, aptright, back, jdict = self.extract(fixed_object, jdict, coordinate)
        if dojsons:
            self.json_handler(tname, 'w', jdict)
        del jdict
        reducplot = self.axesobj[7].imshow(resid, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                                           extent=(coordinate - config.stripewidth // 4,
                                                   coordinate + 1 + config.stripewidth // 4,
                                                   self.pixlow, self.pixhigh))
        plt.colorbar(reducplot, ax=self.axesobj[7])
        aptleftdiff = aptcent - aptleft
        aptrightdiff = aptright - aptcent
        aptcent = aptcent - config.stripewidth // 2 + coordinate + 1
        aptleft = aptcent - aptleftdiff
        aptright = aptcent + aptrightdiff
        self.axesobj[7].plot(aptleft, pixel, color='black', lw=1, ls='--')
        self.axesobj[7].plot(aptright, pixel, color='black', lw=1, ls='--')
        self.axesobj[7].plot(aptcent, pixel, color='black', lw=1, ls='-')
        error = self.poisson(photons)  # creating the errors
        self.axesobj[8].errorbar(pixel, photons + back, yerr=error, color='green')
        self.axesobj[8].plot(pixel, back, color='orange')
        # wavelength calibrate
        wave = self.arc_solution(pixel, aptcent, self.ptoarcs)
        self.axesobj[9].errorbar(wave, photons, yerr=error)
        # flux calibrate
        wave, flux, error, tname = self.calibrate_real(wave, photons, error, tname,
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
                     f'{self.ob}_{self.resolution}_{self.prog}_{self.standard_name}.txt',
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

    def fig_formatter(self):
        """
        Formats the produced reduction figures
        """
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
            for ax in axes[:8]:
                ax.set_xlabel('X Pixel [pix]')
                ax.set_ylabel('Y Pixel [pix]')
            axes[8].set_xlabel('Y Pixel [pix]')
            for ax in axes[8:10]:
                ax.set_ylabel('Counts')
                ax.set_yscale('log')
            axes[9].set_yscale('linear')
            axes[9].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesobj[10].set_title('Final')
        self.axesobj[10].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesobj[10].set_ylabel(r'$F_{\lambda}\ [\mathrm{erg}\ \mathrm{cm}^{-1}\ s^{-1}\ \AA^{-1}]$')
        self.axesstd[10].set_title('Wavelength Calibrated')
        self.axesstd[10].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesstd[10].set_ylabel('Counts')
        self.axesstd[10].set_yscale('linear')
        for ax in (self.axesstdarc, self.axesstd[12], self.axesobjarc, self.axesobj[12]):
            ax.set_xlabel('Y Pixel [pix]')
            # ax.set_ylim(0, 65535)
        for ax in (self.axesobjarc, self.axesstdarc):
            ax.set_ylabel(r'$\Delta \lambda\ [\AA]$')
            ax.legend()
        for ax in (self.axesstd[12], self.axesobj[12]):
            ax.set_ylabel(r'$\lambda\ [\AA]$')
        self.axesobj[13].set_title('Calibration Function')
        self.axesobj[13].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesobj[13].set_ylabel(r'$\mathrm{Counts}\ F_{\lambda}^{-1}\ [\mathrm{erg}^{-1}\ \mathrm{cm}\ s\ \AA]$')
        self.axesstd[11].set_title('Final')
        self.axesstd[11].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesstd[11].set_ylabel(r'$F_{\lambda}\ [\mathrm{erg}\ \mathrm{cm}^{-1}\ s^{-1}\ \AA^{-1}]$')
        self.axesstd[13].set_title('Model')
        self.axesstd[13].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesstd[13].set_ylabel(r'$F_{\lambda}\ [\mathrm{erg}\ \mathrm{cm}^{-1}\ s^{-1}\ \AA^{-1}]$')
        self.axesstd[13].set_yscale('linear')
        self.axesstd[14].set_title('Calibration Function')
        self.axesstd[14].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesstd[14].set_ylabel(r'$\mathrm{Counts}\ F_{\lambda}^{-1}\ [\mathrm{erg}^{-1}\ \mathrm{cm}\ s\ \AA]$')
        [self.figobj.delaxes(self.axesobj[i]) for i in [11, 14, 15]]
        [self.figstd.delaxes(self.axesstd[i]) for i in [15, ]]
        for fig in (self.figobj, self.figstd):
            fig.tight_layout(pad=1.5)
        objfname = f'{self.ob}_{self.resolution}_{self.prog}_{self.target}.png'
        stdfname = f'{self.ob}_{self.resolution}_{self.prog}_{self.standard_name}_for_{self.target}.png'
        for fname in (objfname, stdfname):
            fnameback = fname[:fname.find('.png')] + '.bak.png'
            try:
                os.rename(f'alt_redspec/reduction/{fname}', f'alt_redspec/reduction/{fnameback}')
            except (FileNotFoundError, OSError):
                pass  # if it's not there already, whatever
        self.figobj.savefig(f'alt_redspec/reduction/{objfname}', bbox_inches='tight')
        self.figstd.savefig(f'alt_redspec/reduction/{stdfname}', bbox_inches='tight')
        self.figobjarc.savefig(f'alt_redspec/arcs/{objfname}', bbox_inches='tight')
        self.figstdarc.savefig(f'alt_redspec/arcs/{stdfname}', bbox_inches='tight')
        return

    def logger(self, s: str, w: bool = False):
        """
        Writes log file

        Parameters
        ----------
        w : bool
            Whether to open file or not
        s : str
            String to be written
        """
        if w:
            perm = 'w+'
        else:
            perm = 'a+'
        with open(f'alt_redspec/log/{self.resolution}_{self.prog}_{self.ob}', perm) as f:
            f.write(s + '\n')
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


class Config:
    """
    Config file parameters for use in reduction
    """
    def __init__(self, conf_fname: str):
        """
        Create all parameters

        Parameters
        ----------
        conf_fname: str
            The filename of the plain text config file
        """
        self.fname = conf_fname
        self.allparams = self.fopen()
        self.rawpath, self.redpath, self.targetlist,\
        self.minpix, self.maxpix, self.stripewidth, self.cpix,\
        self.minwave, self.maxwave = self.getparams()
        return

    def getparams(self):
        """
        Grabs all parameters from the dictionary

        Returns
        -------
        rawpath: str
            Path to raw data to be reduced
        redpath: str
            Path to storage folder
        targetlist: str
            The name of the file with added data (e.g. exact target names, else taken from header)
        minpix: int
            The minimum pixel to extract on the ccd
        maxpix: int
            The maximum pixel to extract on the ccd
        stripewidth: int
            The width of pixels down the dispersion axis
        cpix: int
            The typical central pixel
        minwave: int
            Minimum wavelength in Angstroms
        maxwave: int
            Maximum wavelength in Angstroms
        """
        rawpath, redpath, targetlist, minpix, maxpix,\
        stripewidth, cpix,\
        minwave, maxwave = '', '', '', 1, 2051, 100, 250, 5000, 10000
        for key in self.allparams:
            if key == 'rawpath':
                rawpath = self.allparams[key]
                if rawpath == '' and \
                        np.any([not glob.glob(folder) for folder in ['arc', 'bias', 'flat', 'stds', 'object']]):
                    raise ValueError('Using default value for raw data path, it is empty')
            elif key == 'redpath':
                redpath = self.allparams[key]
                if redpath == '':
                    warnings.warn('Using default value (current dir) for reduced data path')
            elif key == 'targetlist':
                targetlist = self.allparams[key]
                if targetlist == '':
                    warnings.warn('No target list given, will default to header values')
            elif key == 'minpix':
                try:
                    minpix = int(self.allparams[key])
                except ValueError:
                    warnings.warn('Using default value for min pixel (1)')
            elif key == 'maxpix':
                try:
                    maxpix = int(self.allparams[key])
                except ValueError:
                    warnings.warn('Using default value for max pixel (2051)')
            elif key == 'stripewidth':
                try:
                    stripewidth = int(self.allparams[key])
                except ValueError:
                    warnings.warn('Using default value for stripe width (100)')
            elif key == 'cpix':
                try:
                    cpix = int(self.allparams[key])
                except ValueError:
                    warnings.warn('Using default value for central pixel (250)')
            elif key == 'minwave':
                try:
                    minwave = int(self.allparams[key])
                except ValueError:
                    warnings.warn('Using default value for minimum wavelength (5000A)')
            elif key == 'maxwave':
                try:
                    maxwave = int(self.allparams[key])
                except ValueError:
                    warnings.warn('Using default value for maximum wavelength (10000A)')
            else:
                warnings.warn(f'Unknown key {key}')
        return rawpath, redpath, targetlist, minpix, maxpix, stripewidth, cpix, minwave, maxwave

    def fopen(self):
        """
        Opens the config file, will skip first line

        Returns
        -------
        d: dict
            All the variables
        """
        d, delim = {}, ':'
        with open(self.fname, 'r') as f:
            for i, line in enumerate(f):
                linesplit, delim = self.what_delim(line.rstrip('\n'), delim)
                if not i:
                    continue
                key = linesplit[0].strip()
                value = linesplit[1].strip()
                d[key] = value
        return d

    @staticmethod
    def what_delim(s: str, fg: str = ':') -> Tuple[Sequence, str]:
        """
        Tries a few different delimiters to split a line

        Parameters
        ----------
        s: str
            The string to be split on
        fg: str
            The first guess of the delimiter

        Returns
        -------
        sl: Sequence
            The split list
        delim: str
            The delimiter that worked on this row
        """
        delims = {':', '#', '\t', ' ', ',', '='}
        sl = s.split(fg)
        try:
            if not len(sl) - 1:  # not split properly
                raise IndexError
        except IndexError:
            for delim in delims.difference(fg):
                sl = s.split(delim)
                if len(sl) == 2:
                    break
            else:
                raise KeyError(f'Cannot parse config file correctly, expecting delimiters from: {delims}')
        else:
            delim = fg
        return sl, delim


def main():
    """Main control module

    All classes are controlled from here. It checks required files are present and accounted for.

    Raises
    ------
    FileNotFoundError
        If a required file is not where it should be (normally current directory)
    """

    # preamble
    t0 = time.time()  # start a clock timer
    np.seterr(divide='ignore', invalid='ignore')  # numpy provides a warning for zero division, ignore this
    warnings.simplefilter("ignore", category=RuntimeWarning)  # another warning is for runtime, ignore this
    warnings.simplefilter('ignore', np.RankWarning)  # another warning about poorly fitting polynomial, ignore
    warnings.simplefilter('ignore', OptimizeWarning)
    ob_list = glob.glob(config.rawpath)
    redpth = config.redpath

    # check for required directories and files
    if 'alt_done.log' not in os.listdir('.'):
        with open('alt_done.log', 'w+'):  # empty the log file
            pass
    if not glob.glob(redpth):
        warnings.warn(f'Could not find {redpth}, attempting to create')
        os.mkdir(redpth)
    for folder in ('arcs', 'bias', 'calib_funcs', 'flat',
                   'jsons', 'log', 'objects', 'reduction', 'residuals', 'standards'):
        if not glob.glob(f'{redpth}/{folder}'):
            warnings.warn(f'Trying to make folder in {redpth}')
            os.mkdir(f'{redpth}/{folder}')

    # check BPM and make otherwise
    if len(glob.glob('BPM_*_python.txt')) != 2:
        print('No bad pixel masks present, creating now from all flats per resolution.')
        bpm = BPM()
        bpm.make_bpm(f'{config.rawpath}/flat/0*fits')  # make the BPM
        print('Made bad pixel mask.')

    # prompt user if they want to repeat all reductions
    do_all = args.do_all
    do_repeat = args.repeat
    if do_all:
        if len(glob.glob(f'{redpth}/**/*txt')):
            os.system(f'rm {redpth}/**/*txt')  # delete the spectra
        with open('alt_done.log', 'w+'):  # empty the log file
            pass
    if not do_repeat:
        # checking which files have already been reduced
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

    # thread the unreduced files
    if len(ob_list):
        avail_cores = 4 or 1  # available cores to thread over
        if len(ob_list) < avail_cores:
            avail_cores = len(ob_list)
        print(f'Threading over {avail_cores} cores.')
        pool = multiprocessing.Pool(processes=avail_cores)
        pool.map(OB, ob_list)
        pool.close()
        print('Done with spectra.')
        print(f'Run took {round((time.time() - t0) / 60, 1)} minutes.')
    else:
        print(f'Process took {round(time.time() - t0, 1)} seconds.')

    print(f'Total time taken was {round((time.time() - t0) / 60, 1)} minutes.')
    return


if __name__ == '__main__':  # if called as script, run main module
    imgnorm = LogNorm(1, 65536)
    ccd_bincmap = LinearSegmentedColormap.from_list('bincmap', plt.cm.binary_r(np.linspace(0, 1, 2)), N=2)
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    myargs.add_argument('-a', '--do-all', action='store_true', default=False, help='Do all spectra?')
    myargs.add_argument('-r', '--repeat', action='store_true', default=False, help='Re-do already reducted spectra?')
    myargs.add_argument('-c', '--config-file', help='The config file', required=True)
    myargs.add_argument('-j', '--gen-jsons', action='store_true', default=False, help='Create jsons for live plots?')
    args = myargs.parse_args()
    dojsons = args.gen_jsons
    config = Config(args.config_file)
    rc('text', usetex=True)
    rcParams.update({'axes.labelsize': 'small', 'axes.titlesize': 'small',
                     'xtick.labelsize': 'small', 'ytick.labelsize': 'small',
                     'legend.fontsize': 'small', 'font.serif': ['Helvetica', 'Arial',
                                                                'Tahoma', 'Lucida Grande',
                                                                'DejaVu Sans'],
                     'font.family': 'serif', 'legend.frameon': False, 'legend.facecolor': 'none',
                     'mathtext.fontset': 'cm', 'mathtext.default': 'regular',
                     'figure.figsize': [4, 3], 'figure.dpi': 144,
                     'xtick.top': True, 'ytick.right': True, 'legend.handletextpad': 0.5,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True})
    main()
