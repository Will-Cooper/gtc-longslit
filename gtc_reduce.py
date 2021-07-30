"""
A Python-3 script to reduce GTC OSIRIS longslit optical spectra.

It is designed to work over multiple processors and will require a directory for the reduced spectra to be put in,
this is the redpath argument in the config file.

It will search for raw spectra in the config rawpath argument (without ending slash) in which we expect:
'resolution/programme_ID/observing_block/' inside each observing block directory we also expect:
'bias/', 'flat/', 'object/', 'arc/' and 'stds/' inside which are the fits files beginning with '0'.

Config parameters include:
rawpath : str
    Path to the raw spectra
redpath : str
    Path to the reduced spectra
targetlist : str (optional)
    Name of a file that contains an the header name and actual target name
head_actual : str (optional)
    The column names in said targetlistl, split as head_actual to convert from header name to actual target name
minpix : int (pixel)
    The minimum pixel on the dispersion axis to reduce within
maxpix : int (pixel)
    The maximum pixel on the dispersion axis to reduce within
stripewidth : int (pixel)
    The width in pixels over which to determine background/ find the source (larger=better but beware of shifts)
spatialminpix: int (pixel)
    The minimum pixel on the spatial axis to get data from
spatialmaxpix: int (pixel)
    The maximum pixel on the spatial axis to get the data from
cpix : int (pixel)
    The central pixel one could typically find the spectra (not used in actual extraction)
minwave : int (Angstroms)
    The mimimum wavelength of the grism (used to cut the line list)
maxwave : int (Angstroms)
    The maximum wavelength of the grism (used to cut the line list)
maxthread : int
    The number of threads to use multiprocessing on
initsolution : str
    The filepath to the initial wavelength solution text file, with columns (wave, pixel)
tracelinestart : int
    The pixel to start the trace solution from, within which the spectra should not diverge by more than 11 pixels
extractmethod : str
    'simple' or 'optimal' as extraction methods

Required file:
    * <name>.config  -- file containing config arguments, see example

Classes
-------
OB : str
    The full class, passed the string to the observing block and hence reduces the spectra in that observing block
BPM : str
    Determines the bad pixel mask if the corresponding resolution mask does not exist in the current environment
Config : str
    Parses the config file for the required arguments: rawpath, redpath, targetlist, head_actual
     minpix, maxpix, stripewidth, cpix, minwave, maxwave, maxthread
"""
from astropy.io.fits import getdata, getheader
from astropy.convolution import convolve, Box1DKernel
from astropy.modeling import models
from astropy.table import Table  # opening files as data tables
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib import use as backend
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np  # general mathematics and array handling
from numpy.polynomial.polynomial import Polynomial as Poly
from numpy.polynomial.chebyshev import Chebyshev as Cheb
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline as Spline3
from scipy.signal import sawtooth
from specutils import Spectrum1D, conf
from specutils.fitting import fit_lines
from tqdm import tqdm

import argparse
import cProfile
import glob  # equivalent to linux 'ls'
import json
import multiprocessing  # used to overcome the inherent python GIL
import os
import time  # timing processing
from traceback import print_tb
from typing import Tuple, Sequence  # used for type hinting
import warnings  # used for suppressing an annoying warning message about runtime


class OB:
    """Reduces the optical spectra

    This is the primary class of this script which self constructs using a string UNIX path to the observing block
    which contains the spectra.

    First, the data is median stacked (bias subtracting each exposure) off the second CCD before being flat fielded
    and bad pixel masked.

    The aperture width is determined from extracting the standard star before being applied to the object spectra.
    The background is the modal value of the config file stripewidth number of pixels around the aperture.

    Arcs are measured to determine the wavelength calibration, which is also applied to the spectra.

    The extracted, wavelength calibrated standard is divided by its corresponding model spectra (F_lambda)
    to create the flux calibration. This is then applied to the final object spectra.
    """
    # initialise all class attributes (overridden by instance attributes)
    figobj: plt.Figure = None
    axesobj: plt.Axes = None
    figstd: plt.Figure = None
    axesstd: plt.Axes = None
    figobjarc: plt.Figure = None
    axesobjarctop: plt.Axes = None
    axesobjarcbot: plt.Axes = None
    figstdtrace: plt.Figure = None
    axesstdtrace: plt.Axes = None
    axesstdtracenonlin: plt.Axes = None
    figobjtrace: plt.Figure = None
    axesobjtrace: plt.Axes = None
    axesobjtracenonlin: plt.Axes = None
    pixlow: int = 1
    pixhigh: int = 2051
    indlow: int = 0
    indhigh: int = 2051
    minspat: int = 0
    maxspat: int = 1024
    spatminind: int = 0
    spatmaxind: int = 1024
    ptobias: str = ''
    master_bias: np.ndarray = np.empty(0)
    ptoflats: str = ''
    master_flat: np.ndarray = np.empty(0)
    bpm: np.ndarray = np.empty(0)
    ptoobj: str = ''
    humidity: float = 0
    airmass: float = 0
    mjd: float = 0
    ptostds: str = ''
    master_standard: np.ndarray = np.empty(0)
    ptoarcs: str = ''
    ftoabs: np.ndarray = np.empty(0)
    ftoabs_error: np.ndarray = np.empty(0)
    standard_name: str = ''
    standard_residual: np.ndarray = np.empty(0)
    pbar: tqdm = None
    cpix: int = 100
    aptleft: np.ndarray = np.empty(0)
    aptright: np.ndarray = np.empty(0)
    haveaperture: bool = False
    master_target: np.ndarray = np.empty(0)
    target: str = ''
    target_residual: np.ndarray = np.empty(0)
    wave_soln: Poly = None
    gain: float = 1
    tracelinestart: int = 1287
    extractmethod: str = 'simple'
    geoshift: pd.DataFrame = pd.DataFrame()

    def __init__(self, ptodata: str):
        """
        Parameters
        ----------
        ptodata : str
            The UNIX path as a string to where the observing block is
        """
        tproc0 = time.time()
        self.ob: str = ptodata.split('/')[-1]  # observing block of this spectra
        self.prog: str = ptodata.split('/')[2]  # program ID
        self.resolution: str = ptodata.split('/')[1]  # resolution of this spectra
        if np.any([folder not in os.listdir(ptodata) for folder in ['arc', 'bias', 'flat', 'stds', 'object']]):
            print(f'Cannot reduce {self.resolution} {self.prog} {self.ob} due to missing directories')
            return
        self.logger(f'Resolution {self.resolution}\nProgramme {self.prog}\nObserving block {self.ob}', w=True)
        try:
            self.reduction(ptodata)
        except (ValueError, AttributeError, FileNotFoundError, IndexError, KeyError) as e:
            try:
                self.pbar.close()
            except TypeError:
                pass
            print(f'Could not complete reduction of {self.ob} {self.prog} {self.resolution}; see below:')
            print_tb(e.__traceback__)
            print(e)
            return
        else:
            print(f'Object processed: {self.target} for {self.resolution} '
                  f'grism in {self.ob} with walltime {round(time.time() - tproc0, 1)} seconds.')
        return

    def reduction(self, ptodata: str):
        # initialising
        self.pbar = tqdm(total=100, desc=f'{self.resolution}/{self.prog}/{self.ob}')
        self.figobj, self.axesobj = plt.subplots(4, 4, figsize=(16, 10), dpi=300)
        self.axesobj: np.array(plt.Axes) = self.axesobj.flatten()
        self.figstd, self.axesstd = plt.subplots(4, 4, figsize=(16, 10), dpi=300)
        self.axesstd: np.array(plt.Axes) = self.axesstd.flatten()
        self.figobjarc: plt.Figure = plt.figure(figsize=(8, 5), dpi=300)
        self.axesobjarctop: plt.Axes = self.figobjarc.add_axes([0.1, 0.35, 0.8, 0.55])
        self.axesobjarcbot: plt.Axes = self.figobjarc.add_axes([0.1, 0.1, 0.8, 0.25])
        self.figstdtrace, (self.axesstdtrace, self.axesstdtracenonlin) = plt.subplots(nrows=2, sharex=True,
                                                                                      figsize=(8, 5), dpi=300)
        self.figobjtrace, (self.axesobjtrace, self.axesobjtracenonlin) = plt.subplots(nrows=2, sharex=True,
                                                                                      figsize=(8, 5), dpi=300)
        # pixel limits
        self.pixlow, self.pixhigh,\
        self.indlow, self.indhigh,\
        self.spatminind, self.spatmaxind = self.pixel_constraints()
        self.minspat, self.maxspat = self.spatial_constraints()
        self.tracelinestart = config.tracelinestart - self.pixlow - 1
        self.pbar.update(5)
        self.logger(f'Use pixels from {self.pixlow} to {self.pixhigh}')
        # bad pixel mask
        self.bpm = self.get_bpm()
        # bias
        self.ptobias = ptodata + '/bias/0*fits'  # path to biases
        self.logger(f'There are {len(glob.glob(self.ptobias))} bias files')
        self.master_bias = self.bias()  # creates the master bias file
        self.pbar.update(5)
        biasplot = self.axesobj[4].imshow(self.master_bias, cmap='BuPu', origin='lower', aspect='auto',
                                          extent=(config.spatialminpix, config.spatialmaxpix,
                                                  self.pixlow, self.pixhigh))
        plt.colorbar(biasplot, ax=self.axesobj[4])
        # flat
        self.ptoflats = ptodata + '/flat/0*fits'  # path to flats
        self.logger(f'There are {len(glob.glob(self.ptoflats))} flats files')
        self.master_flat = self.flat(self.master_bias)  # creates the master flat file
        self.pbar.update(5)
        flatplot = self.axesobj[5].imshow(self.master_flat, cmap='BuPu', origin='lower', aspect='auto',
                                          extent=(config.spatialminpix, config.spatialmaxpix,
                                                  self.pixlow, self.pixhigh))
        plt.colorbar(flatplot, ax=self.axesobj[5])
        # wave solution
        self.ptoarcs = ptodata + '/arc/0*fits'  # path to arcs
        self.logger(f'There are {len(glob.glob(self.ptoarcs))} arcs files')
        if not os.path.exists(config.geocorrect):
            self.geoshift = self.geometric_distortion(self.ptoarcs)
        else:
            self.geoshift = pd.read_csv(config.geocorrect, names=np.arange(self.spatminind, self.spatmaxind + 1))
            self.geoshift.set_index(np.arange(self.pixlow, self.pixhigh + 1), inplace=True)
        self.wave_soln = self.arc_solution(self.ptoarcs)
        self.pbar.update(5)
        bpmplot = self.axesobj[6].imshow(self.bpm, cmap=ccd_bincmap, origin='lower', aspect='auto',
                                         extent=(config.spatialminpix, config.spatialmaxpix,
                                                 self.pixlow, self.pixhigh))
        plt.colorbar(bpmplot, ax=self.axesobj[6])
        # header info and further initialisation
        self.ptoobj = ptodata + '/object/0*fits'  # path to object
        self.logger(f'There are {len(glob.glob(self.ptoobj))} object files')
        self.humidity, self.airmass, self.mjd = self.hum_air()  # average humidity and airmass of object obs
        self.pbar.update(5)
        self.logger(f'Humidity {self.humidity}\nAirmass {self.airmass}\nMJD {self.mjd}')
        self.ptostds = ptodata + '/stds/0*scopy.fits'  # path to standard
        self.logger(f'There are {len(glob.glob(self.ptostds))} standards files')
        # standard
        self.logger('The standard is being reduced and analysed:')
        self.haveaperture = False
        self.ftoabs, self.master_standard, self.ftoabs_error, self.standard_name, \
        self.standard_residual, self.cpix, self.aptleft, self.aptright = self.standard()  # standard reduction
        self.haveaperture = True
        self.pbar.update(5)
        self.logger(f'Name of standard being used was {self.standard_name.upper()}')
        # object
        self.logger('The object is now being reduced:')
        self.master_target, self.target, self.target_residual = self.object()  # reduces target
        self.pbar.update(5)
        # writing files and creating plots
        self.logger(f'The target was {self.target}')
        self.fig_formatter()  # formats the plots (titles, labels)
        self.writing()  # writes reduced spectra to files
        self.pbar.update(5)
        self.pbar.close()
        del self.pbar
        return

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
        colnames = config.head_actual.split('_')
        try:
            tinfo = Table.read(config.targetlist)  # table containing header names and shortnames
            if np.any([col not in tinfo.colnames for col in colnames]):
                raise ValueError
        except (OSError, FileNotFoundError, ValueError):
            pass
        else:
            for row in tinfo:
                if row[colnames[0]].strip() == sname:  # if the header names match
                    sname = row[colnames[1]]  # take the shortname as the target name
                    break
        return sname

    def get_pixcoord(self, data: np.ndarray) -> int:
        """Gets the pixel coordinate

        The row pixel where the spectra is normally, worked out from the median across the whole array
        """
        data: np.ndarray = data[:, self.minspat: self.maxspat]
        offset: int = config.spatialminpix + self.minspat - 1
        coord = np.nanmedian(np.argmax(data, axis=-1)).astype(int)
        return coord + offset  # average central pixel

    @staticmethod
    def spatial_constraints() -> Tuple[int, int]:
        """
        Finds spatial axis index constraints

        Returns
        -------
        minspat: int
            The minimum index to slice array on spatial axis (close to center)
        maxspat: int
            The maximum index to slice array on spatial axis (close to center)
        """
        minspat: int = config.cpix - config.spatialminpix - config.stripewidth // 2 - 1
        maxspat: int = config.cpix - config.spatialminpix + config.stripewidth // 2
        return minspat, maxspat

    @staticmethod
    def pixel_constraints() -> Tuple[int, int, int, int, int, int]:
        """Finds the constraints for the respective resolution

        Use the config minpix and maxpix values to get the limits of extraction in pixel and index space
        """
        xmin, xmax = config.minpix, config.maxpix
        indmin = xmin - 1
        indmax = xmax  # max index = max pixel to be inclusive (like iraf)
        spatmin, spatmax = config.spatialminpix - 1, config.spatialmaxpix
        return xmin, xmax, indmin, indmax, spatmin, spatmax

    @staticmethod
    def bisub(data: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Bias subtract from each CCD given

        Elementwise subtraction of the entire CCD

        Parameters
        ----------
        bias : np.ndarray
            The bias from same observing block as the object
        data : np.ndarray
            The full CCD that will have the bias subtracted
        """
        return np.subtract(data, bias)

    def get_bpm(self):
        """
        Loads the bad pixel mask

        Returns
        -------
        mask: np.ndarray
            The bad pixel mask (array of 0 is bad, 1 is good)
        """
        mask: np.ndarray = np.loadtxt(f'BPM_{self.resolution}_python.txt')
        return self.region_trim(mask)

    def bpm_applying(self, data: np.ndarray) -> np.ndarray:
        """Applies the bad pixel mask

        Elementwise multiplication where 0 is a bad pixel and 1 (unchanged) for good pixels

        Parameters
        ----------
        data : np.ndarray
            The full CCD that needs to be masked
        """
        mask: np.ndarray = self.bpm
        return np.multiply(mask, data)  # BPM is array of 1=good, 0=bad

    def fopener(self, fname: str, dotrim: bool = True) -> np.ndarray:
        """Opens the fits file using astropy

        Parameters
        ----------
        fname : str
            The relative or full string to the file to be opened
        dotrim : bool
            Switch to trim or not
        """
        data: np.ndarray = getdata(fname, ext=2)
        if dotrim:
            data = self.region_trim(data)
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

    def bias(self) -> np.ndarray:
        """Creates master bias

        All the biases are taken from the fits files in the UNIX path given in __init__. They are then 3D stacked
        and then unstacked along the median back to a 2D CCD.
        """
        bias_list = glob.glob(self.ptobias)  # list all the biases
        all_bias = np.stack([self.fopener(bias) for bias in bias_list])  # creates an array of the bias CCDs
        median_bias = self.med_stack(all_bias)  # median stacks the biases
        return self.bpm_applying(median_bias)  # apply bad pixel mask

    def region_trim(self, data: np.ndarray) -> np.ndarray:
        """
        Downsizes an array on the config specifications (pixel limits)

        Parameters
        ----------
        data: np.ndarray
            The full size array to be trimmed

        Returns
        -------
        dtrimmed: np.ndarray
            The trimmed array
        """
        if np.any(np.less([self.indlow, self.spatminind], 0)) \
        or np.any(np.greater([self.indhigh, self.spatmaxind], len(data))):
            raise ValueError('Check indices, not slicing data correctly')
        dtrimmed = data[self.indlow: self.indhigh, self.spatminind: self.spatmaxind]
        return dtrimmed

    @staticmethod
    def normalise(data: np.ndarray) -> np.ndarray:
        """Normalises to the mean value of array

        Take the median value of the approximate cut-out of the spectra to normalise by

        Parameters
        ----------
        data : np.ndarray
            The full CCD of the median flat
        """
        return np.divide(data, np.median(data))

    def flat(self, bias: np.ndarray) -> np.ndarray:
        """Creates master flat

        All the flats are taken from the fits files in the UNIX path given in __init__. They are then 3D stacked
        and then unstacked along the median back to a 2D CCD. They then have the bad pixel mask applied and
        are finally normalised to the spectral region.

        Parameters
        ----------
        bias : np.ndarray
            The full CCD of the bias file
        """
        flat_list = glob.glob(self.ptoflats)  # list of all flats in observing block
        all_flats = np.stack([self.fopenbisub(_flat, bias) for _flat in flat_list])
        median_flat = self.med_stack(all_flats)  # determines bias subtracted median flat
        bpm_flat = self.bpm_applying(median_flat)  # apply bad pixel mask
        return self.normalise(bpm_flat)  # normalised flat

    def geometric_distortion(self, ptoarcs: str) -> pd.DataFrame:
        cind: int = config.cpix - self.spatminind - 1
        pixel: np.ndarray = np.arange(self.pixlow, self.pixhigh + 1)
        arcfiles = glob.glob(ptoarcs)
        xup: np.ndarray = np.arange(cind, self.spatmaxind - self.spatminind, 20)
        xdown: np.ndarray = np.arange(cind, self.spatminind, -20)
        spatial: np.ndarray = np.arange(self.spatminind, self.spatmaxind + 1)
        shifts = pd.DataFrame(data={xval: np.zeros(pixel.shape) for xval in spatial}, index=pixel)
        initsol = None
        for i, arc in tqdm(enumerate(arcfiles), total=len(arcfiles), leave=None, desc='Analysing Arcs'):
            xy = {}
            arcdata = self.fopenbisub(arc, self.master_bias)  # extract whole arc
            arcdata = self.flat_field(arcdata, self.master_flat)
            lamp = getheader(arc, ext=0)['OBJECT'].split('_')[-1].lower()
            for xarr in (xup, xdown):
                for k, x in tqdm(enumerate(xarr), total=len(xarr), leave=None, desc='Identifying'):
                    if not k:
                        initsol = None
                    arccut: np.ndarray = arcdata[:, x]  # just central pixel stripe
                    pix_wave = self.identify(pixel, arccut, lamp, initsol, True)
                    initsol = pix_wave
                    if len(pix_wave):
                        xy[x] = pix_wave.pixel.values
            xydf = pd.DataFrame(data=xy)
            xydf = xydf.reindex(sorted(xydf.columns), axis=1)
            for tracenum, trace in tqdm(xydf.iterrows(), total=len(xydf), leave=None, desc='Finding shifts'):
                tracecheb: Cheb = Cheb.fit(trace.index.values, trace.values, deg=6, full=True)[0]  # x to y
                traceyvals: np.ndarray = tracecheb(spatial)
                yints = traceyvals.astype(int)
                diffs: np.ndarray = traceyvals[cind] - traceyvals
                for k, xval in tqdm(enumerate(spatial), total=len(spatial), leave=None, desc='Assigning shifts'):
                    shifts[xval][yints[k]] = diffs[k]
        for k, xval in tqdm(enumerate(spatial), total=len(spatial), leave=None, desc='Assigning shifts'):
            shiftxcol: pd.Series = shifts[xval].copy()
            shiftxcol = shiftxcol[np.logical_not(np.isclose(shiftxcol, 0))]
            try:
                ycheb: Cheb = Cheb.fit(shiftxcol.index.values, shiftxcol.values, deg=6, full=True)[0]  # y to shift
            except (np.linalg.LinAlgError, ValueError):
                shifts[xval] = shifts[xval - 1]
                continue
            diffs: np.ndarray = ycheb(pixel)
            for kk, yval in tqdm(enumerate(pixel), total=len(pixel), leave=None, desc='Assigning shifts'):
                shifts[xval][yval] = diffs[kk]
        shifts.to_csv('geoshifttest.csv', header=False, index=False)
        return shifts

    def transform(self, data: np.ndarray) -> np.ndarray:
        newdata: np.ndarray = np.zeros_like(data)
        datalen: int = len(data)
        for i, spatrow in tqdm(enumerate(data), total=datalen, desc='Transforming', leave=None):
            for j, val in enumerate(spatrow):
                shift: float = self.geoshift[j + self.spatminind + 1][i + self.pixlow]
                sign: int = np.sign(shift).astype(int)
                shiftind: int = np.floor(shift).astype(int)
                idiff: int = i + shiftind
                newi: int = idiff if 0 <= idiff < datalen else None
                if newi is None:
                    continue
                nexti: int = idiff + sign if 0 <= idiff + sign < datalen else None
                nextiremainder: float = shift - shiftind
                newiremainder: float = 1 - nextiremainder
                newdata[newi, j] += val * newiremainder
                if nexti is not None:
                    newdata[nexti, j] += val * nextiremainder
        return newdata

    @staticmethod
    def identify(pixel: np.ndarray, dataline: np.ndarray, lamp: str,
                 initsol: pd.DataFrame = None, fastsolve: bool = False) -> pd.DataFrame:
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
        initsol: pd.DataFrame
            The initial solution to start looking around, if None use file
        fastsolve: bool
            Switch to do a simple solver just to find central pixel

        Returns
        -------
        pix_wave: pd.DataFrame
            A dataframe of columns pixel to wavelength
        """
        if initsol is None:
            initsolution: pd.DataFrame = pd.read_csv(config.initsolution)
            initsolution: pd.DataFrame = initsolution[[i in lamp for i in initsolution.line]].copy()
        else:
            initsolution = initsol
        spline = Spline3(pixel, dataline)
        strengths = [np.max([i for i in spline(np.linspace(val - 2, val + 2))]) for val in initsolution.pixel]
        initsolution['strength'] = strengths
        xhigh = np.linspace(np.min(pixel), np.max(pixel), len(pixel) * 100)
        yfit = spline(xhigh)
        if fastsolve:
            ypos = [xhigh[np.argmax([i for i in spline(np.linspace(val - 5, val + 6, 100))]) + int(val - 5) * 100]
                    for val in initsolution.pixel]
            initsolution.pixel = ypos
            return initsolution[['pixel', 'wave']]
        spectrum = Spectrum1D(spectral_axis=xhigh * u.pixel, flux=yfit * u.count)
        g_inits = []
        for i, val in initsolution.iterrows():
            g_inits.append(models.Gaussian1D(amplitude=val.strength * u.count,
                                             mean=val.pixel * u.pixel,
                                             stddev=1.5 * u.pixel,
                                             bounds={'stddev': (1, 2),
                                                     'mean': ((val.pixel - 4) * u.pixel,
                                                              (val.pixel + 4) * u.pixel)}))
        g_initfull = np.sum(g_inits)
        g_fits = fit_lines(spectrum, g_initfull, window=8 * u.pixel)
        if len(g_inits) > 1:
            initsolution['newpixel'] = [val.mean.value for val in g_fits]
        else:
            initsolution['newpixel'] = g_fits.mean.value
        initsolution.pixel = initsolution.newpixel
        return initsolution[['pixel', 'wave']]

    def solution_fitter(self, df: pd.DataFrame, ax: Sequence[plt.Axes]) -> Poly:
        """
        Iteratively fits to arc solution

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe of pixel to wavelength
        ax : Sequence[plt.Axes]
            The plot axes to put the residuals on

        Returns
        -------
        pcomb: Poly
            The combined polynomial solution
        """
        def dofits():
            _pcomb: Cheb = Cheb.fit(df.pixel.values, df.wave.values, deg=1, full=True)[0]
            _residual: np.ndarray = _pcomb(df.pixel.values) - df.wave.values
            axtop.plot(df.pixel, _residual, color=cols[i], lw=0, marker='x')
            chebfit: Cheb = Cheb.fit(df.pixel.values, _residual, deg=len(df) // 2, full=True)[0]
            axtop.plot(*chebfit.linspace(), color=cols[i], ls='--', label=f'Residual N={len(df)}')
            _pcomb: Cheb = _pcomb - chebfit
            _residual: np.ndarray = _pcomb(df.pixel.values) - df.wave.values
            axbot.plot(df.pixel, _residual, color=cols[i], marker='x', lw=0)
            df['residual'] = _residual
            _df = df[np.abs(df.residual) < 1].copy()
            # chebresfit: Cheb = Cheb.fit(_df.pixel.values, _df.residual.values, deg=4, full=True)[0]
            # axbot.plot(*chebresfit.linspace(), ls='--', color=cols[i])
            # _pcomb -= chebresfit
            _residual = _pcomb(_df.pixel.values) - _df.wave.values
            return _pcomb, _residual, _df
        axtop, axbot = ax
        axtop: plt.Axes = axtop
        axbot: plt.Axes = axbot
        cols: list = ['tab:' + i for i in ('cyan', 'blue', 'purple', 'pink', 'orange', 'red')]
        i: int = 0
        pcomb: Cheb = Cheb.fit(df.pixel.values, df.wave.values, deg=1, full=True)[0]
        residual: np.ndarray = pcomb(df.pixel.values) - df.wave.values
        while i < 2:
            pcomb, residual, df = dofits()
            residual = np.abs(residual)
            df = df[residual < 3 * np.std(residual)].copy()
            i += 1
        iqr: np.ndarray = np.subtract(*np.quantile(residual, [.75, .25]))
        rms: float = np.sqrt(np.sum(np.square(residual)) / len(residual))
        rmsd: float = np.sqrt(np.sum(np.square(0 - residual)) / len(residual))
        rmsdiqr: float = rmsd / iqr
        self.logger(f'RMS = {rms} for {len(residual)} points')
        self.logger(f'Arc solution RMSDIQR {rmsdiqr}')
        return pcomb

    def arc_solution(self, ptoarcs: str) -> Poly:
        """
        Generates the wavelength solution from the arcs

        Parameters
        ----------
        ptoarcs: str
            The path to the directory holding the arc files

        Returns
        -------
        soln: Poly
            The wavelength solution
        """
        # TODO: make arc solution work for R300R
        cpix: int = config.cpix - config.spatialminpix
        pixel: np.ndarray = np.arange(self.pixlow, self.pixhigh + 1)
        axmain = self.axesobj[12], self.axesstd[12]
        ax2 = self.axesobjarctop, self.axesobjarcbot
        arcfiles = glob.glob(ptoarcs)
        all_pix_wave = pd.DataFrame(columns=('pixel', 'wave'))
        for i, arc in enumerate(arcfiles):
            arcdata = self.fopenbisub(arc, self.master_bias)  # extract whole arc
            arcdata = self.flat_field(arcdata, self.master_flat)
            arccut: np.ndarray = arcdata[:, cpix - 1]  # just central pixel stripe
            lamp = getheader(arc, ext=0)['OBJECT'].split('_')[-1].lower()
            pix_wave = self.identify(pixel, arccut, lamp)
            all_pix_wave = all_pix_wave.append(pix_wave).reset_index(drop=True)
        all_pix_wave.sort_values('pixel', inplace=True, ignore_index=True)
        # arcdata = getdata('stuart/ftb_arcs_use_wav_cal.fits')
        # arccut = arcdata[:, cpix - 1].flatten()
        # all_pix_wave = self.identify(pixel, arccut, 'all', usegiven=True)
        # all_pix_wave.sort_values('pixel', inplace=True, ignore_index=True)
        soln = self.solution_fitter(all_pix_wave, ax2)
        for ax in axmain:
            ax.plot(all_pix_wave.pixel, all_pix_wave.wave, 'kx')
        return soln

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
            if head['GRISM'].strip() == 'OPEN':  # i.e. acquisiton
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
        return np.divide(data, flat, where=np.logical_not(np.isclose(flat, 0)))  # returns 0 on the bad pixels

    @staticmethod
    def find_back(segment: np.ndarray, cpix: float, xsmall: np.ndarray) -> Tuple[np.ndarray, Cheb, float]:
        """
        Finds the background value using an iterative mode

        Parameters
        ----------
        segment: np.ndarray
            Aperture axis cut off
        cpix: float
            The previous row aperture centre
        xsmall: np.ndarray
            The array of pixel indices

        Returns
        -------
        background: np.ndarray
            The vector background count representing the sky
        chebfit: Chebyshev
            The chebyshev fit for the background
        variance: float
            The variance on pixel by pixel basis
        """
        leftmin = np.floor(cpix - 49).astype(int) if cpix - 49 > 0 else 0  # left region start
        leftmax = np.floor(cpix - 31).astype(int) if cpix - 30 > 0 else leftmin + 19  # left region end
        rightmax = np.ceil(cpix + 39).astype(int) if cpix + 40 < len(segment) else len(segment)  # right region end
        rightmin = np.ceil(cpix + 19).astype(int) if cpix + 19 < len(segment) else rightmax - 21  # right region start
        reg1: np.ndarray = segment[leftmin: leftmax]  # median count of left region
        reg2: np.ndarray = segment[rightmin: rightmax]  # median count of right region
        # reg1 = convolve(reg1, Box1DKernel(2))
        # reg2 = convolve(reg2, Box1DKernel(2))
        ind1: np.ndarray = np.median(xsmall[leftmin: leftmax])  # median index of left region
        ind2: np.ndarray = np.median(xsmall[rightmin: rightmax])  # median index of right region
        backmed: np.ndarray = np.array([np.median(reg1), np.median(reg2)])  # the two background medians
        variance: np.ndarray = np.var(np.hstack([reg1, reg2]))
        indmed: np.ndarray = np.array([ind1, ind2])  # the two index medians
        chebfit: Cheb = Cheb.fit(indmed, backmed, deg=1, full=True)[0]  # linear fit between regions
        background: np.ndarray = chebfit(xsmall)  # the background on a pixel by pixel basis
        return background, chebfit, variance

    def peak_average(self, segment: np.ndarray, cpix: float, ind: int, jdict: dict) \
            -> Tuple[float, int, int, int, dict]:
        """Takes the strip of the CCD and gets the median around the peak

        This method extracts the full signal using 3/4 HWHM.

        Parameters
        ----------
        segment : np.ndarray
            The row around the spectral pixel center
        cpix : float
            The central pixel value
        ind : int
            Index along dispersion axis
        jdict : dict
            Dictionary of extraction results
        """
        jdict[ind] = thisdict = {}
        xsmall = np.arange(len(segment), dtype=float)
        background, backfit, variance = self.find_back(segment, cpix, xsmall)
        leftwidth, rightwidth = 2.5, 2.5
        minind = np.floor(cpix - leftwidth).astype(int) if cpix - leftwidth >= 0 else 0
        maxind = np.ceil(cpix + rightwidth).astype(int) if cpix + rightwidth <= len(segment) else len(segment)
        backsub = np.subtract(segment, background, out=np.zeros_like(segment),
                              where=np.greater(segment, background))
        # backsub = np.subtract(segment, background)
        # backsub = np.divide(backsub, variance)
        minreg = np.min(backsub)
        xbig = np.linspace(0, len(backsub), 1000, endpoint=False, dtype=float)
        amp = np.nanmax(backsub[minind: maxind]).astype(float)
        thisdict['xsmall'] = xsmall.tolist()
        thisdict['xbig'] = xbig.tolist()
        yvals = np.interp(xbig, xsmall, backsub)
        bigback = backfit(xbig)
        thisdict['region'] = segment.astype(float).tolist()
        thisdict['ybig'] = (yvals + bigback).astype(float).tolist()
        if self.extractmethod == 'optimal':
            p0 = [amp, cpix, 2]
            try:
                gfit = curve_fit(self.gaussian, xbig, yvals, p0=p0,
                                 bounds=([0, minind, .5], [1.2 * amp, maxind, 6]))[0]
            except (ValueError, RuntimeError):
                gfit = p0
            yvals_gauss = self.gaussian(xbig, *gfit)
            ysmall = self.gaussian(xsmall, *gfit)
            amp = gfit[0]
            bigslice: np.ndarray = np.flatnonzero(yvals_gauss > 0.2 * amp)
            smallslice: np.ndarray = np.flatnonzero(ysmall > 0.2 * amp)
            signal: float = np.sum(backsub[smallslice])
            bigcpix_minind, bigcpix_maxind = xbig[bigslice][[0, -1]]
            cpix_minind, cpix_maxind = xsmall[smallslice][[0, -1]]
        elif self.extractmethod == 'simple':
            cpix_minind = cpix - 5 if cpix - 5 > 0 else 0
            cpix_maxind = cpix + 6 if cpix + 6 < len(backsub) else len(backsub)
            bigcpix_minind: int = np.flatnonzero(xbig > cpix_minind)[0]
            bigcpix_maxind: int = np.flatnonzero(xbig > cpix_maxind)[0]
            cpix_minind: int = np.flatnonzero(xsmall > cpix_minind)[0]
            cpix_maxind: int = np.flatnonzero(xsmall > cpix_maxind)[0]
            backsubregion: np.ndarray = convolve(backsub[cpix_minind: cpix_maxind], Box1DKernel(2))
            signal: float = np.sum(backsubregion)
            yvals_gauss = yvals
        else:
            raise ValueError('Invalid extraction method given, must be of "simple", "optimal"')
        backcpix = backfit(cpix)
        thisdict['yfit'] = (yvals_gauss + bigback).astype(float).tolist()
        thisdict['background'] = bigback.astype(float).tolist()
        thisdict['params'] = [float(xbig[bigcpix_minind]), float(cpix), float(xbig[bigcpix_maxind]),
                              float(amp) + float(backcpix),
                              float(minreg)]

        return signal, cpix_minind, cpix_maxind, backcpix, jdict

    @staticmethod
    def center1d(initial: float, data: np.ndarray, npts: int = 101,
                 width: float = 11, radius: float = 11, interpfact: int = 10):
        def smallinds():
            _minind = int(interpfact * (xval - halfwidth)) if xval - halfwidth > 0 else 0
            _maxind = int(interpfact * (xval + halfwidth)) if xval + halfwidth < npts else bignpts
            return _minind, _maxind
        x: np.ndarray = np.arange(npts)
        bignpts: int = int(interpfact * npts)
        xbig: np.ndarray = np.linspace(x[0], x[-1], bignpts)
        databig: np.ndarray = np.interp(xbig, x, data)
        halfwidth: float = width / 2
        b: float = radius + 1.5 * width
        bigminind: int = int(interpfact * (initial - b)) if initial - b > 0 else 0
        bigmaxind: int = int(interpfact * (initial + b)) if initial + b < npts else bignpts
        xcut: np.ndarray = xbig[bigminind: bigmaxind]
        integrated = np.empty_like(xcut)
        for i, xval in enumerate(xcut):
            minind, maxind = smallinds()
            integrated[i] = np.trapz(databig[minind: maxind] * sawtooth(xbig[minind: maxind] - xval))
        mididx: int = (np.argmin(integrated) - np.argmax(integrated)) // 2 + np.argmax(integrated)
        cpix: float = xcut[mididx]
        # idxs = argrelmax(databig, order=int(width * interpfact))[0]
        # idx = idxs[np.argmax(databig[idxs])]
        # wind = windows.gaussian(npts, halfwidth)
        # filtered = convolve(databig, wind, mode='same') / np.sum(wind)
        # idx = np.argmax(filtered)
        # cpix = xbig[idx]
        return cpix

    def trace(self, data: np.ndarray):
        def centerspatial(i: int, cpix: float) -> Tuple[float, float]:
            minind: int = i - 5 if i - 5 > 0 else 0  # step size down
            maxind: int = i + 6 if i + 6 < len(data) else len(data)  # step size up
            linedata: np.ndarray = np.median(data[minind: maxind], axis=0)  # median stacked data in range
            # yvals: np.ndarray = np.interp(spatialbig, spatial, linedata)
            amp: float = np.max(linedata)  # guess amplitude
            if cpix is None:  # first pixel
                cpix = np.argmax(linedata)  # guess center
            # medcent: float = self.center1d(cpix, linedata, len(linedata))
            minind: int = cpix - 11 if cpix - 11 > 0 else 0  # width to look for line down
            maxind: int = cpix + 11 if cpix + 11 < len(spatial) else len(spatial)  # width to look for line up
            p0 = [amp, cpix, 2]
            try:
                gfit = curve_fit(self.gaussian, spatial, linedata, p0=p0,
                                 bounds=([0, minind, .5], [1.2 * amp, maxind, 6]))[0]
            except (ValueError, RuntimeError):  # failed fit
                gfit = p0  # default to input parameters
            medcent: float = gfit[1]  # fit center
            return medcent

        if self.haveaperture:
            (ax1, ax2) = (self.axesobjtrace, self.axesobjtracenonlin)
        else:
            (ax1, ax2) = (self.axesstdtrace, self.axesstdtracenonlin)
        firstline: int = self.tracelinestart
        spatial: np.ndarray = np.arange(len(data[0]))
        # spatialbig: np.ndarray = np.linspace(spatial.min(), spatial.max(), len(spatial) * 10)
        firstpix = centerspatial(firstline, None)
        dcent = {firstline: firstpix}
        stepdownlines: np.ndarray = np.arange(firstline, 0, -5)[1:]
        stepuplines: np.ndarray = np.arange(firstline, len(data), 5)[1:]
        for steparr in (stepdownlines, stepuplines):
            initialpix = firstpix
            for line in tqdm(steparr, total=len(steparr), leave=None, desc='Tracing'):
                linecent = centerspatial(line, initialpix)
                if linecent is None:
                    continue
                dcent[line] = linecent
                initialpix = linecent
        dfcent = pd.DataFrame([dcent, ]).T
        dfcent.rename(columns={0: 'pixel'}, inplace=True)
        dfcent.sort_index(inplace=True)
        spcent: Poly = Poly.fit(dfcent.index.values, dfcent.pixel.values, deg=3, full=True)[0]
        ax2.plot(dfcent.index, dfcent.pixel + self.minspat, 'rx')
        spcentx, spcenty = spcent.linspace()
        ax2.plot(spcentx, spcenty + self.minspat, 'r--')
        residual: np.ndarray = spcent(dfcent.index.values) - dfcent.pixel.values
        iqr: np.ndarray = np.subtract(*np.quantile(residual, [.75, .25]))
        dfcent['residual'] = residual
        ax1.plot(dfcent.index, dfcent.residual, 'rx', label=f'Initial, N={len(dfcent)}')
        [ax1.axhline(iqrval, linestyle='--', color='r') for iqrval in (-iqr, iqr)]
        dfcentcut = dfcent[np.abs(dfcent.residual) < iqr].copy()
        if len(dfcentcut) > len(dfcent) // 2:
            spcent: Poly = Poly.fit(dfcentcut.index.values, dfcentcut.pixel.values, deg=3, full=True)[0]
            dfcent = dfcentcut
        ax2.plot(dfcent.index, dfcent.pixel + self.minspat, 'bx')
        spcentx, spcenty = spcent.linspace()
        ax2.plot(spcentx, spcenty + self.minspat, 'b--')
        residual: np.ndarray = spcent(dfcent.index.values) - dfcent.pixel.values
        iqr: np.ndarray = np.subtract(*np.quantile(residual, [.75, .25]))
        rms: float = np.sqrt(np.sum(np.square(residual)) / len(residual))
        rmsd: float = np.sqrt(np.sum(np.square(0 - residual)) / len(residual))
        rmsdiqr: float = rmsd / iqr
        dfcent['residual'] = residual
        ax1.plot(dfcent.index, dfcent.residual, 'bx', label=f'Iterated, N={len(dfcent)}')
        [ax1.axhline(iqrval, linestyle='--', color='b') for iqrval in (-iqr, iqr)]
        ax1.legend()
        self.logger(f'Trace RMS: {rms:.3f}, RMSIQR: {rmsdiqr:.3f} for N={len(residual)}')
        return spcent

    def extract(self, data: np.ndarray, jdict: dict) -> Tuple[np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray,
                                                              np.ndarray, dict]:
        """Extracts the spectrum

        Take a slice around the central pixel with a sub-section of rows selected from the CCD.
        Extract and background subtract the signal from that slice on a row by row basis.

        Parameters
        ----------
        data : np.ndarray
            The full CCD of the observation, to be sliced and extracted from
        jdict : dict
            Dictionary of extraction results
        """
        data: np.ndarray = data[:, self.minspat: self.maxspat]  # slicing spectra
        pixels: np.ndarray = np.arange(self.pixlow, self.pixhigh + 1)
        spcent: Poly = self.trace(data)
        peaks: np.ndarray = np.empty_like(pixels, dtype=float)
        aptleft: np.ndarray = np.empty_like(pixels, dtype=float)
        aptcent: np.ndarray = spcent(pixels - 1)
        aptright: np.ndarray = np.empty_like(pixels, dtype=float)
        background: np.ndarray = np.empty_like(pixels, dtype=float)
        for i, row in tqdm(enumerate(data), total=len(data), desc='Extracting', leave=None):
            cpix = aptcent[i]
            peak_extract = self.peak_average(row, cpix, i, jdict)
            peaks[i] = peak_extract[0]
            aptleft[i] = peak_extract[1]
            aptright[i] = peak_extract[2]
            background[i] = peak_extract[3]
            jdict = peak_extract[4]
        peaks *= self.gain
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
        # TODO: this isn't how errors work
        return np.divide(np.sqrt(photon_count), photon_count, where=np.logical_not(np.isclose(photon_count, 0)))

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
        return np.sqrt(np.divide(top, bottom, where=np.logical_not(np.isclose(bottom, 0))))

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
    def get_header_info(fname: str) -> Tuple[str, float, float, float, float]:
        """Gets the standard file name

        Parameters
        ----------
        fname : str
            The UNIX path to the file being opened as a string
        """
        head = getheader(fname, ext=0)  # the observational information on OSIRIS is on the first HDU
        return head['OBJECT'].rstrip(), head['HUMIDITY'], head['AIRMASS'], head['MJD-OBS'],\
            head['SLITW'], head['GAIN']

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
        with open(f'{config.redpath}/jsons/{self.ob}_{self.resolution}_{self.prog}_{objname}.json', perm) as jfile:
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
        biasplot = self.axesstd[4].imshow(self.master_bias, cmap='BuPu', origin='lower', aspect='auto',
                                          extent=(config.spatialminpix, config.spatialmaxpix,
                                                  self.pixlow, self.pixhigh))
        plt.colorbar(biasplot, ax=self.axesstd[4])
        flatplot = self.axesstd[5].imshow(self.master_flat, cmap='BuPu', origin='lower', aspect='auto',
                                          extent=(config.spatialminpix, config.spatialmaxpix,
                                                  self.pixlow, self.pixhigh))
        plt.colorbar(flatplot, ax=self.axesstd[5])
        bpmplot = self.axesstd[6].imshow(self.bpm, cmap=ccd_bincmap, origin='lower', aspect='auto',
                                         extent=(config.spatialminpix, config.spatialmaxpix,
                                                 self.pixlow, self.pixhigh))
        plt.colorbar(bpmplot, ax=self.axesstd[6])
        standard_list = self.checkifspectra(glob.glob(self.ptostds))  # list of standards)
        sname = self.get_header_info(standard_list[-1])[0]  # gets name of standard
        self.gain = self.get_header_info(standard_list[-1])[-1]
        sname = sname.split('_')[-1].lower().replace('-', '')  # converting standard name from header to model name
        slitwidths = [self.get_header_info(f)[-2] for f in standard_list]
        self.logger(f'Standard has slit widths: {slitwidths}')
        self.pbar.update(5)
        # stack raw data
        all_standards = np.stack([self.fopener(obj) for obj in standard_list])
        median_standard = self.med_stack(all_standards)  # median stack objects
        self.axesstd[0].imshow(median_standard, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                               extent=(config.spatialminpix, config.spatialmaxpix,
                                       self.pixlow, self.pixhigh))
        self.axesstd[0].set_title(f'Median Stack ({len(all_standards)} standard/s)')
        self.pbar.update(5)
        # bias subtract
        all_standards = np.stack([self.bisub(std, self.master_bias) for std in all_standards])
        median_standard = self.med_stack(all_standards)  # median stack the bias subtracted standards
        stdcoord = self.get_pixcoord(median_standard)  # centre pixel
        self.logger(f'Central pixel for standard around {stdcoord + 1}')
        self.axesstd[1].imshow(median_standard, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                               extent=(config.spatialminpix, config.spatialmaxpix,
                                       self.pixlow, self.pixhigh))
        self.pbar.update(5)
        # flat field
        flat_standard = self.flat_field(median_standard, self.master_flat)  # flat field the standard
        self.axesstd[2].imshow(flat_standard, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                               extent=(config.spatialminpix, config.spatialmaxpix,
                                       self.pixlow, self.pixhigh))
        # apply bad pixel mask & geo correct
        fixed_standard = self.transform(flat_standard)
        self.axesstd[3].imshow(fixed_standard, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                               extent=(config.spatialminpix, config.spatialmaxpix,
                                       self.pixlow, self.pixhigh))
        self.pbar.update(5)
        # extract spectra
        # fixed_standard = getdata('stuart/tr_iftb_Ross640_std_ccd2.fits')
        jdict = {}
        if dojsons:
            self.json_handler(sname, 'w', {})
            jdict = self.json_handler(sname, 'r')
        pixel, photons, resid, aptleft, aptcent, aptright, back, jdict = self.extract(fixed_standard, jdict)
        if dojsons:
            self.json_handler(sname, 'w', jdict)
        del jdict
        outcpix = aptcent
        aptleftdiff = aptcent - aptleft
        aptrightdiff = aptright - aptcent
        aptcent = aptcent + config.spatialminpix + self.minspat
        aptleft = aptcent - aptleftdiff
        aptright = aptcent + aptrightdiff
        reducplot = self.axesstd[7].imshow(resid, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                                           extent=(config.spatialminpix + self.minspat,
                                                   config.spatialminpix + self.maxspat,
                                                   self.pixlow, self.pixhigh))
        plt.colorbar(reducplot, ax=self.axesstd[7])
        self.axesstd[7].plot(aptleft, pixel, color='black', lw=1, ls='--')
        self.axesstd[7].plot(aptright, pixel, color='black', lw=1, ls='--')
        self.axesstd[7].plot(aptcent, pixel, color='black', lw=1, ls='-')
        error = self.poisson(photons)  # creating the errors
        self.axesstd[8].errorbar(pixel, photons + back, yerr=error, color='green')
        self.axesstd[8].plot(pixel, back, color='orange')
        self.pbar.update(5)
        # wavelength calibrate
        wave: np.ndarray = self.wave_soln(pixel)
        self.axesstd[12].plot(pixel, wave, 'k--')
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
        self.pbar.update(5)
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
        flux = np.divide(photons, ftoabs, where=np.logical_not(np.isclose(ftoabs, 0)))  # flux at the Earth
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
            _hum, _air, _mjd = self.get_header_info(fname)[1:-2]
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
        self.gain = self.get_header_info(object_list[-1])[-1]
        slitwidths = [self.get_header_info(f)[-1] for f in object_list]
        self.logger(f'Object has slit widths: {slitwidths}')
        # raw data
        all_objects = np.stack([self.fopener(obj) for obj in object_list])
        median_object = self.med_stack(all_objects)  # median stack objects
        self.axesobj[0].imshow(median_object, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                               extent=(config.spatialminpix, config.spatialmaxpix,
                                       self.pixlow, self.pixhigh))
        self.axesobj[0].set_title(f'Median Stack ({len(all_objects)} object/s)')
        self.pbar.update(5)
        # centre coordinate
        coordinate = self.get_pixcoord(median_object)  # find pixel coordinate
        self.logger(f'Central pixel for object around {coordinate + 1}')
        # bias subtract
        all_objects = np.stack([self.bisub(obj, self.master_bias) for obj in all_objects])
        median_object = self.med_stack(all_objects)  # median stack bias subtracted objects
        self.axesobj[1].imshow(median_object, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                               extent=(config.spatialminpix, config.spatialmaxpix,
                                       self.pixlow, self.pixhigh))
        self.pbar.update(5)
        # flat field
        flat_object = self.flat_field(median_object, self.master_flat)  # flat field the object
        self.axesobj[2].imshow(flat_object, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                               extent=(config.spatialminpix, config.spatialmaxpix,
                                       self.pixlow, self.pixhigh))
        # apply bad pixel mask & geo correct
        fixed_object = self.transform(flat_object)
        self.axesobj[3].imshow(fixed_object, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                               extent=(config.spatialminpix, config.spatialmaxpix,
                                       self.pixlow, self.pixhigh))
        self.pbar.update(5)
        # extract spectra
        # fixed_object = getdata('stuart/tr_iftb_J1745m1640_com.fits')
        jdict = {}
        if dojsons:
            self.json_handler(tname, 'w', {})
            jdict = self.json_handler(tname, 'r')
        pixel, photons, resid,\
        aptleft, aptcent, aptright, back, jdict = self.extract(fixed_object, jdict)
        if dojsons:
            self.json_handler(tname, 'w', jdict)
        del jdict
        reducplot = self.axesobj[7].imshow(resid, cmap='coolwarm', norm=imgnorm, origin='lower', aspect='auto',
                                           extent=(config.spatialminpix + self.minspat - 1,
                                                   config.spatialminpix + self.maxspat,
                                                   self.pixlow, self.pixhigh))
        plt.colorbar(reducplot, ax=self.axesobj[7])
        aptleftdiff = aptcent - aptleft
        aptrightdiff = aptright - aptcent
        aptcent = aptcent + config.spatialminpix + self.minspat
        aptleft = aptcent - aptleftdiff
        aptright = aptcent + aptrightdiff
        self.axesobj[7].plot(aptleft, pixel, color='black', lw=1, ls='--')
        self.axesobj[7].plot(aptright, pixel, color='black', lw=1, ls='--')
        self.axesobj[7].plot(aptcent, pixel, color='black', lw=1, ls='-')
        error = self.poisson(photons)  # creating the errors
        self.axesobj[8].errorbar(pixel, photons + back, yerr=error, color='green')
        self.axesobj[8].plot(pixel, back, color='orange')
        self.pbar.update(5)
        # wavelength calibrate
        wave: np.ndarray = self.wave_soln(pixel)
        self.axesobj[12].plot(pixel, wave, 'k--')
        self.axesobj[9].errorbar(wave, photons, yerr=error)
        # flux calibrate
        wave, flux, error, tname = self.calibrate_real(wave, photons, error, tname,
                                                       self.ftoabs, self.ftoabs_error)  # real units spectra
        self.pbar.update(5)
        self.axesobj[10].errorbar(wave, flux, yerr=error)
        self.axesobj[13].errorbar(wave, self.ftoabs, yerr=self.ftoabs_error, color='red')
        calib_spectra = np.array((wave, flux, error))
        return calib_spectra, tname, resid

    def writing(self):
        """
        Writes the standard used and object to files
        """
        filename = f'{config.redpath}/npdata/{self.ob}_{self.resolution}_{self.prog}_{self.target}'
        if do_backup and os.path.exists(filename + '.npz'):
            os.rename(filename + '.npz', filename + '_bak.npz')
        np.savez_compressed(filename,
                            target=self.master_target.T, standard=self.master_standard.T,
                            bias=self.master_bias, flat=self.master_flat,
                            cutout=self.target_residual, cutoutstd=self.standard_residual,
                            fluxcal=np.array((self.master_standard[0], self.ftoabs)).T)
        with open('reduced.log', 'a+') as f:  # add to log file that the target has been reduced
            f.write(f'{self.ob}_{self.resolution}_{self.prog}\n')
        return

    def fig_formatter(self):
        """
        Formats the produced reduction figures
        """
        self.axesobj: Sequence[plt.Axes] = self.axesobj
        self.axesstd: Sequence[plt.Axes] = self.axesstd
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
            self.pbar.update(1)
        self.axesobj[10].set_title('Final')
        self.axesobj[10].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesobj[10].set_ylabel(r'$F_{\lambda}\ [\mathrm{erg}\ \mathrm{cm}^{-1}\ s^{-1}\ \AA^{-1}]$')
        self.axesstd[10].set_title('Wavelength Calibrated')
        self.axesstd[10].set_xlabel(r'$\lambda\ [\AA]$')
        self.axesstd[10].set_ylabel('Counts')
        self.axesstd[10].set_yscale('linear')
        self.axesstdtrace.set_title('Standard Trace')
        self.axesstdtracenonlin.set_xlabel('Y Pixel [pix]')
        self.axesstdtrace.set_ylabel(r'$\Delta$ X Pixel [pix]')
        self.axesstdtracenonlin.set_ylabel('X Pixel [pix]')
        self.axesobjtrace.set_title('Object Trace')
        self.axesobjtracenonlin.set_xlabel('Y Pixel [pix]')
        self.axesobjtrace.set_ylabel(r'$\Delta$ X Pixel [pix]')
        self.axesobjtracenonlin.set_ylabel('X Pixel [pix]')
        for ax in (self.axesstd[12],
                   self.axesobjarctop, self.axesobjarcbot, self.axesobj[12]):
            ax.set_xlabel('Y Pixel [pix]')
        for ax in (self.axesobjarctop, ):
            ax.set_ylabel(r'$\Delta \lambda\ [\AA]$')
            ax.legend(ncol=6)
            ax.set_xlim(np.floor(config.minpix / 100) * 100, np.ceil(config.maxpix / 100) * 100)
        for ax in (self.axesobjarcbot, ):
            ax.set_ylabel(r'$\Delta \lambda\ [\AA]$')
            ax.set_ylim(-.015, .015)
            ax.set_xlim(np.floor(config.minpix / 100) * 100, np.ceil(config.maxpix / 100) * 100)
            ax.set_yticks([-.01, 0, .01])
        for ax in (self.axesstd[12], self.axesobj[12]):
            ax.set_ylabel(r'$\lambda\ [\AA]$')
        self.pbar.update(1)
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
        self.pbar.update(1)
        for fig in (self.figobj, self.figstd):
            fig.tight_layout(pad=1.5)
        objfname = f'{self.ob}_{self.resolution}_{self.prog}_{self.target}.png'
        stdfname = f'{self.ob}_{self.resolution}_{self.prog}_{self.standard_name}_for_{self.target}.png'
        if do_backup:
            for fname in (objfname, stdfname):
                fnameback = fname[:fname.find('.png')] + '.bak.png'
                if os.path.exists(f'{config.redpath}/reduction/{fname}'):
                    os.rename(f'{config.redpath}/reduction/{fname}', f'{config.redpath}/reduction/{fnameback}')
        self.figobj.savefig(f'{config.redpath}/reduction/{objfname}', bbox_inches='tight')
        self.figstd.savefig(f'{config.redpath}/reduction/{stdfname}', bbox_inches='tight')
        self.figobjarc.savefig(f'{config.redpath}/arcs/{objfname}', bbox_inches='tight')
        self.figstdtrace.savefig(f'{config.redpath}/trace/{stdfname}', bbox_inches='tight')
        self.figobjtrace.savefig(f'{config.redpath}/trace/{objfname}', bbox_inches='tight')
        self.pbar.update(1)
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
        with open(f'{config.redpath}/log/{self.resolution}_{self.prog}_{self.ob}', perm) as f:
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
        all_flats = np.stack([OB.fopener(flat, False) for flat in flat_list])  # stack the CCDs in the 3rd dimension
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
    rawpath: str = ''
    redpath: str = ''
    targetlist: str = ''
    head_actual: str = ''
    minpix: int = 1
    maxpix: int = 2051
    spatialminpix: int = 1
    spatialmaxpix: int = 1024
    stripewidth: int = 100
    cpix: int = 250
    minwave: int = 1e3
    maxwave: int = 1e5
    maxthread: int = multiprocessing.cpu_count() // 2
    initsolution: str = ''
    tracelinestart: int = 1292
    extractmethod: str = 'simple'
    geocorrect: str = ''

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
        self.getparams()
        return

    def getparams(self):
        """
        Grabs all parameters from the dictionary
        """
        for key in self.allparams:
            if key == 'rawpath':
                self.rawpath = self.allparams[key]
                if self.rawpath == '' and \
                        np.any([not glob.glob(folder) for folder in ['arc', 'bias', 'flat', 'stds', 'object']]):
                    raise ValueError('Using default value for raw data path, it is empty')
            elif key == 'redpath':
                self.redpath = self.allparams[key]
                if self.redpath == '':
                    warnings.warn('Using default value (current dir) for reduced data path')
            elif key == 'targetlist':
                self.targetlist = self.allparams[key]
                if self.targetlist == '':
                    warnings.warn('No target list given, will default to header values')
            elif key == 'head_actual':
                self.head_actual = self.allparams[key]
                if self.head_actual == '' or len(self.head_actual.split('_')) != 2:
                    warnings.warn('No column names given in for targetlist to use or cannot split on "_"')
            elif key == 'minpix':
                try:
                    self.minpix = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for min pixel {self.maxpix}')
            elif key == 'maxpix':
                try:
                    self.maxpix = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for max pixel {self.minpix}')
            elif key == 'stripewidth':
                try:
                    self.stripewidth = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for stripe width {self.stripewidth}')
            elif key == 'cpix':
                try:
                    self.cpix = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for central pixel {self.cpix}')
            elif key == 'minwave':
                try:
                    self.minwave = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for minimum wavelength {self.minwave}A')
            elif key == 'maxwave':
                try:
                    self.maxwave = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for maximum wavelength {self.maxwave}A')
            elif key == 'maxthread':
                try:
                    self.maxthread = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for max threads {self.maxthread}')
            elif key == 'spatialminpix':
                try:
                    self.spatialminpix = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for min spatial axis pixel {self.spatialminpix}')
            elif key == 'spatialmaxpix':
                try:
                    self.spatialmaxpix = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for max spatial axis pixel {self.spatialmaxpix}')
            elif key == 'initsolution':
                self.initsolution = str(self.allparams[key])
                if self.initsolution == '' or not len(glob.glob(self.initsolution)):
                    raise ValueError('No file path given to initial wavelength solution or cannot find file')
            elif key == 'tracelinestart':
                try:
                    self.tracelinestart = int(self.allparams[key])
                except ValueError:
                    warnings.warn(f'Using default value for trace line start {self.tracelinestart}')
            elif key == 'extractmethod':
                self.extractmethod = str(self.allparams[key])
                if self.extractmethod == '':
                    warnings.warn(f'Using default value for extraction method "simple"')
                    self.extractmethod = 'simple'
            elif key == 'geocorrect':
                self.geocorrect = str(self.allparams[key])
                if not os.path.exists(self.geocorrect):
                    warnings.warn(f'Calculating geometric correction (takes time)')
            else:
                warnings.warn(f'Unknown key {key}')
        return

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


def env_check():
    """
    Checks the environment for required directories and files, will attempt to fix otherwise
    """
    redpth = config.redpath

    # check for required directories and files
    if 'reduced.log' not in os.listdir('.'):
        with open('reduced.log', 'w+'):  # empty the log file
            pass
    if not glob.glob(redpth):
        warnings.warn(f'Could not find {redpth}, attempting to create')
        os.mkdir(redpth)
    for folder in ('arcs', 'bias', 'calib_funcs', 'flat', 'standards',
                   'jsons', 'log', 'objects', 'reduction', 'npdata', 'trace',
                   'residuals', 'residuals/objects', 'residuals/standards'):
        if not glob.glob(f'{redpth}/{folder}'):
            warnings.warn(f'Trying to make folder in {redpth}')
            os.mkdir(f'{redpth}/{folder}')

    # check BPM and make otherwise
    if len(glob.glob('BPM_*_python.txt')) != 2:
        print('No bad pixel masks present, creating now from all flats per resolution.')
        bpm = BPM()
        bpm.make_bpm(f'{config.rawpath}/flat/0*fits')  # make the BPM
        print('Made bad pixel mask.')
    return


def create_reduction_list():
    """

    Returns
    -------
    ob_list : list
        List of observing blocks to be reduced
    """
    ob_list = glob.glob(config.rawpath)

    if do_all:
        with open('reduced.log', 'w+'):  # empty the log file
            pass
    if not do_repeat:
        # checking which files have already been reduced
        ob_list = np.array(ob_list)
        done_list = np.array([], dtype=bool)
        for obs in ob_list:
            with open('reduced.log', 'r') as f:
                for line in f:
                    if obs.split('/')[-1] in line and obs.split('/')[1] in line and obs.split('/')[2] in line:
                        done_list = np.append(done_list, False)
                        break
                else:
                    done_list = np.append(done_list, True)
        ob_list = ob_list[done_list]
    return ob_list


def run_reduction(ob_list: list):
    """
    Runs the reduction across available threads

    Parameters
    ----------
    ob_list: list
        The list of observing blocks to be reduced
    """
    t0 = time.time()  # start a clock timer
    # thread the unreduced files
    if len(ob_list):  # if the list isn't empty
        avail_cores = config.maxthread or 1  # available cores to thread over
        if len(ob_list) < avail_cores:
            avail_cores = len(ob_list)
        if len(ob_list) > 1:
            print(f'Threading over {avail_cores} core(s).')
            pool = multiprocessing.Pool(processes=avail_cores)
            pool.map(OB, ob_list)
            pool.close()
        else:
            _ = OB(ob_list[0])
        print('Done with spectra.')
        tfin = (time.time() - t0) / 60
        print(f'Run took {tfin:.1f} minutes.')
    else:
        tfin = time.time() - t0
        print(f'Process took {tfin:.1f} seconds.')
    return


def main():
    """
    Main control module
    """
    env_check()  # check the environment
    ob_list = create_reduction_list()
    run_reduction(ob_list)
    return


def system_arguments():
    """
    Creates the system arguments for the script and parses them when called

    Returns
    -------
    args
        The arguments passed by the user to the script
    """
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    myargs.add_argument('-a', '--do-all', action='store_true', default=False, help='Do all spectra?')
    myargs.add_argument('-r', '--repeat', action='store_true', default=False, help='Re-do already reducted spectra?')
    myargs.add_argument('-c', '--config-file', help='The config file', required=True)
    myargs.add_argument('-j', '--gen-jsons', action='store_true', default=False, help='Create jsons for live plots?')
    myargs.add_argument('-t', '--trace', action='store_true', default=False, help='Trace profile?')
    myargs.add_argument('-b', '--backup', action='store_true', default=False, help='Create backup files when saving?')
    _args = myargs.parse_args()
    return _args


if __name__ == '__main__':  # if called as script, run main module
    # global constants will go here
    print('Compiled.')
    warnings.filterwarnings('ignore', message='The fit may be')
    # plotting
    conf.do_continuum_function_check = False
    backend('agg')
    imgnorm = LogNorm(1, 65536)
    ccd_bincmap = LinearSegmentedColormap.from_list('bincmap', plt.cm.binary_r(np.linspace(0, 1, 2)), N=2)
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
    # system arguments
    args = system_arguments()
    do_all = args.do_all
    do_repeat = args.repeat
    do_trace = args.trace
    do_backup = args.backup
    if do_all and not do_repeat:
        do_repeat = True
    dojsons = args.gen_jsons
    # config file
    config = Config(args.config_file)
    # run the script
    if do_trace:
        cProfile.run('main()')
    else:
        main()
