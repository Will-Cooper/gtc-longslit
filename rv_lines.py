from barycorrpy import get_BC_vel
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import numpy as np
from scipy.interpolate import interp1d

import argparse
from typing import Tuple
from warnings import simplefilter


def region_chopper(wavemin: float, wavemax: float, wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    boolcut = np.logical_and(wave > wavemin, wave < wavemax)
    flux = flux[boolcut]
    return wave[boolcut], flux / np.nanmedian(flux)


def poly_cutter(wave: np.ndarray, flux: np.ndarray, polycoeff: int) -> Tuple[np.ndarray, np.ndarray]:
    p = interp1d(wave, flux, kind=polycoeff)
    x = np.linspace(np.min(wave), np.max(wave), 100)
    y = p(x)
    return x, y


def lowest_within5(x: np.ndarray, y: np.ndarray, midline: float) -> float:
    lowx, highx = midline - 5, midline + 5
    boolcut = np.logical_and(x > lowx, x < highx)
    lowvalind = np.argmin(y[boolcut])
    lowest = x[boolcut][lowvalind]
    return lowest, round(lowest - midline, 2)


def rv_calc(delta_wave: float, lab_wave: float) -> float:
    return 299792458 / 1e3 * delta_wave / lab_wave


def tab_query(fname: str, uncor_rv: float, tabname: str = 'Master_info_correct_cm.csv') -> np.ndarray:
    fname = fname.split('/')[-1]
    fname = fname[:fname.find('.')]
    ob, res, prog, tname = fname.split('_')
    df = pd.read_csv(tabname)
    df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)
    acqcheck = False
    df_cut = df[np.logical_and(tname == df.shortname,
                               np.logical_and(ob == df.ob,
                                              np.logical_and(res == df.resolution,
                                                             np.logical_and(df.acquistion == acqcheck,
                                                                            prog == df.program
                                                                            )
                                                             )
                                              )
                               )
                ].copy()
    if not len(df_cut):
        return None
    df_first = df_cut.iloc[0].copy()
    ra = df_first.ra
    dec = df_first.dec
    pmra = df_first.pmra
    pmdec = df_first.pmdec
    plx = df_first.parallax
    mean_jd = df_cut.mjd.mean() + 2400000.5
    lat = 28.75666667  # of gtc
    long = 17.89194444  # of gtc
    alt = 2300  # of gtc
    zmeas = 0
    rv = uncor_rv
    epoch = 2457189  # gaia dr2 epochal jd
    ephemeris = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de405.bsp'
    barycor = get_BC_vel(JDUTC=mean_jd, ra=ra, dec=dec, pmra=pmra, pmdec=pmdec, px=plx,
                         lat=lat, longi=long, alt=alt, zmeas=zmeas, rv=rv, epoch=epoch,
                         obsname='', ephemeris=ephemeris, leap_update=True)[0]
    return barycor


def line_plotter(wave: np.ndarray, flux: np.ndarray, fname: str):
    fig, axs = plt.subplots(4, 2, figsize=(8, 5), dpi=244)
    axs = axs.flatten()
    colours = ['blue'] * 2 + ['orange'] * 2
    linestyles = ['-', '--'] * 2
    uncor_rv_list = []
    for i, spec_index in enumerate(spec_indices):
        ax = axs[i]
        labline = spec_indices[spec_index]
        wavecut, fluxcut = region_chopper(labline - 25, labline + 25, wave, flux)
        ax.scatter(wavecut, fluxcut, marker='s', s=4, fc='white', ec='black', label='Data')
        ax.axvline(labline, color='grey', ls='--', label='Lab')
        for j, k in enumerate(range(3, 11, 2)):
            fitx, fity = poly_cutter(wavecut, fluxcut, k)
            ax.plot(fitx, fity, label=k, color=colours[j], linestyle=linestyles[j])
        lowest, wavshift = lowest_within5(*poly_cutter(wavecut, fluxcut, 9), midline=labline)
        uncor_rv_list.append(rv_calc(wavshift, labline))
        ax.axvline(lowest, color='black', ls='--', label='Measured')
        ax.text(labline, 1.5, rf'$\Delta \lambda = ${wavshift:.2f}$\AA$',
                {'color': 'black', 'fontsize': 'x-small', 'ha': 'center', 'va': 'center',
                 'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
        ax.set_xlim(labline - 25, labline + 25)
        ax.set_ylim(0.2, 1.75)
        ax.set_title('\t'*2+f'{spec_index.capitalize()}')
        if i in (6, 7):
            ax.set_xlabel(r'Wavelength [$\AA$]')
    uncor_rv = round(np.median(uncor_rv_list), 1)
    uncor_err = round(np.std(uncor_rv_list) / np.sqrt(len(uncor_rv_list)), 1)
    barycor = tab_query(fname, uncor_rv)
    if barycor is not None:
        barycor = barycor[0]
        cor_rv = round((barycor + uncor_rv) / 1e3, 1)
        plt.suptitle(rf'Uncorrected RV: {uncor_rv:.1f}$\pm${uncor_err:.1f} km/s; Corrected RV: {cor_rv:.1f} km/s')
    else:
        plt.suptitle(f'Uncorrected RV: {uncor_rv:.1f}$\pm${uncor_err:.1f} km/s')
    axs[4].set_ylabel('Normalised')
    axs[2].set_ylabel(r'Flux [$F_{\lambda}$]')
    axs[1].legend(loc='upper left', bbox_to_anchor=[1., 1.], title='Fit Order')
    fig.subplots_adjust(hspace=0.75)
    plt.show()
    return


def freader(f: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1))  # load file
    except (OSError, FileNotFoundError) as e:
        raise(e, 'Cannot find given file in: ', f)
    except ValueError:
        wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1), skiprows=1)  # load file
    return wave, flux


def main():
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    myargs.add_argument('-f', '--file-name', required=True, help='File to be plotted', type=str)
    args = myargs.parse_args()
    fname = args.file_name
    wave, flux = freader(fname)
    line_plotter(wave, flux, fname)
    return


if __name__ == '__main__':
    rcParams.update({'axes.labelsize': 'small', 'axes.titlesize': 'small',
                     'xtick.labelsize': 'small', 'ytick.labelsize': 'small',
                     'legend.fontsize': 'small', 'font.serif': ['Helvetica', 'Arial',
                                                                'Tahoma', 'Lucida Grande',
                                                                'DejaVu Sans'],
                     'font.family': 'serif', 'legend.frameon': False, 'legend.facecolor': 'none',
                     'mathtext.fontset': 'cm', 'mathtext.default': 'regular',
                     'figure.figsize': [4, 3], 'figure.dpi': 144, 'lines.linewidth': .75,
                     'xtick.top': True, 'ytick.right': True, 'legend.handletextpad': 0.1,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True})
    rc('text', usetex=True)  # using latex
    spec_indices = {'k1-a': 7664.8991, 'k1-b': 7698.9645,
                    'rb1-a': 7800.27, 'rb1-b': 7947.60,
                    'na1-a': 8183.256, 'na1-b': 8194.824,
                    'cs1-a': 8521.13, 'cs1-b': 8943.47}
    simplefilter('ignore', np.RankWarning)  # a warning about poorly fitting polynomial, ignore
    main()
