"""A Python-3 script to plot GTC OSIRIS longslit optical spectra. Currently only supporting R300R and R2500I.

It is designed to work over multiple processors and will require a folder 'alt_redspec/' containing
'objects/', 'standards/', 'residuals/' and 'calib_funcs/'. On the upper level it will place the plots
made here of the resultant spectra derived in the 'alt_reduce.py' script.

It requires packages: numpy, glob, matplotlib, astropy, sys, multiprocessing, and typing.

Required non script files/ folders in the same directory as this script include:
    * Master_info_correct_cm.csv    -- Containing the filenames, programme IDs, observing blocks, resolutions,
                                       shortnames, distances, spectral type and spectral type number.
    * alt_doplot.log    -- A descriptive file for which spectra should be plotted per observing block

It is designed to be imported into the 'alt_reduce.py' script but also capably works as an independent script.

Methods
-------
tabulate()
    Creates the tables required by the spectral sequence
residual_check(t, name)
    Checks if an object should be plotted or not depending on how good the spectra has been resolved (e.g. low S/N)
split_spec_sequence(n, pp, res, tsubset, tcheck)
    Plots the spectral sequence subsetted by type
spec_sequence(t, res)
    The creation of the spectral sequence, splits them by type and plots
standards(res)
    Plots the standards observed in each observing block against the models (not neccesarily matching the target)
objects(res)
    Plots all the objects observed in each observing block
residuals(res, ttype)
    Plots all the cut out regions around the spectra, coloured by count
calib_functions(res)
    Plots the calibration functions for each standard to object
main()
    The main function of this script controlling the above methods
"""
import numpy as np
import glob
from astropy.table import Table
from splat import Spectrum
from splat.plot import plotSpectrum
import astropy.units as u
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
# import multiprocessing
import sys
from typing import Tuple


def tabulate() -> Tuple[Table, Table]:
    """Makes the tables used in spec sequence

    Opens the listed file, pulls out required columns (observation block, programme ID, resolution, target name,
    spectral type and spectral type number). Sorts on spectral type number and splits over the different resolutions.
    """
    t = Table.read('Master_info_correct_cm.csv')
    path, obj, spt, sptnum, res, ob, prog = [], [], [], [], [], [], []
    for i in glob.glob('alt_redspec/objects/*txt'):
        name = i.split('_')[-1].strip('.txt')   # pull the name from the UNIX path
        for row in t:
            if name == row['SHORTNAME']:
                if row['jdksptnum'] > 0:
                    path.append(i)
                    ob.append(i.split('/')[-1].split('_')[0])  # observing block
                    res.append(i.split('/')[-1].split('_')[1])  # resolution
                    prog.append(i.split('/')[-1].split('_')[-2])  # programme ID
                    obj.append(name)  # target name
                    spt.append(row['truejdkspt'])  # spectral type
                    sptnum.append(row['jdksptnum'])  # spectral type number
                break
    tnew = Table(data=(path, ob, res, prog, obj, spt, sptnum),
                 names=('Path', 'OB', 'Resolution', 'Program', 'Object', 'SpT', 'sptnum'))
    tnew.sort(['sptnum', 'Object'])  # sort on spectral type number
    tnew.write('alt_redspec/alt_plotted_files.txt', format='ascii', overwrite=True)
    rc('text', usetex=True)  # using latex
    tnew300 = tnew[tnew['Resolution'] == 'R0300R']  # new table of just R300R spectra
    tnew2500 = tnew[tnew['Resolution'] == 'R2500I']  # new table of just R2500I spectra
    print('Tables created and sorted, now plotting.')
    return tnew300, tnew2500


def residual_check(t: Table, name: str) -> bool:
    """Checks the residual log if the object should be plotted in spectral sequence

    Parameters
    ---------
    t : astropy.table.table.Table
        The table of objects to be plotted and a boolean
    name : str
        The name of the object
    """
    doplot = False
    for row in t:
        if row['col1'] == name:
            if row['col2']:
                doplot = True
            break
    return doplot


def split_spec_sequence(n: int, pp: PdfPages, res: str, tsubset: Table, tcheck: Table):
    """Plots the split spectral sequences

    Parameters
    ----------
    n : int
        The number figure to be opened
    pp : matplotlib.backends.backend_pdf.PdfPages
        The figure environment where the plots are sent
    res : str
        The resolution of these spectra
    tsubset : astropy.table.table.Table
        The table for just this subset of the sequence
    tcheck : astropy.table.table.Table
        The table of objects to be plotted and a boolean
    """
    c, usedc, whernorm = 1, [], []
    plt.figure(n, figsize=(20, 30))  # make figure
    ax = plt.subplot(111)
    for row in tsubset:  # over all rows in new table
        wave, flux = np.loadtxt(row['Path'], unpack=True, usecols=(0, 1))  # read file
        doplot = residual_check(tcheck, f'{row["OB"]}_{res}_{row["Program"]}')
        # doplot = True
        if doplot:
            wave = wave[flux > 0]  # cut to positive flux
            flux = flux[flux > 0]  # cut to positive flux
            w815 = wave[np.logical_and(wave > 8100, wave < 8200)]
            w815 = np.median(w815)
            whernorm.append(w815)
            f815 = np.median(flux[np.logical_and(wave > 8100, wave < 8200)])
            plt.plot(wave, flux / f815 / c,
                     label=f'Target: {row["Object"]}; SpT: {row["SpT"]}')  # plots
            usedc.append(c)
            c *= 100
    plt.title(f'GTC Spectral Sequence with {res} Resolution; subset {n}')  # title
    plt.xlabel(r'Wavelength $\AA$')  # x label
    plt.ylabel(r'Normalised $F_{\lambda}$ $erg/cm^{2}/s/\AA$ at 8150$\AA$')  # y label
    plt.grid(axis='both')  # grid lines on x axis
    plt.yscale('log')
    plt.plot(whernorm, 1 / np.array(usedc), color='black', linewidth=2, linestyle='dashed')
    if res == 'R2500I':
        plt.xticks(np.arange(7250, 10500, 250))
    else:
        plt.xticks(np.arange(6500, 10500, 500))
        plt.xlim(xmin=6500)
    plt.yticks(1 / np.array(usedc))
    plt.legend(loc='center left', bbox_to_anchor=[1.0, 0.5])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.tick_params(axis='both', top=True, direction='in')
    pp.savefig()
    return


def spec_sequence(t: Table, res: str):
    """Plots the spectral sequence

    Parameters
    ----------
    t : astropy.table.table.Table
        The table of given resolution targets
    res : str
        The resolution of these targets
    """
    resid_t = Table.read('alt_doplot.log', format='ascii.no_header')
    resid_t['col2'] = resid_t['col2'] == 'true'
    pp = PdfPages(f'alt_redspec/gtc_specsequence_{res}.pdf')
    if res == 'R2500I':
        split_spec_sequence(1, pp, res, t[t['sptnum'] < 70], resid_t)
        split_spec_sequence(2, pp, res, t[np.logical_and(t['sptnum'] >= 70, t['sptnum'] < 72)], resid_t)
        split_spec_sequence(3, pp, res, t[t['sptnum'] >= 72], resid_t)
    else:
        split_spec_sequence(1, pp, res, t[t['sptnum'] < 71], resid_t)
        split_spec_sequence(2, pp, res, t[t['sptnum'] >= 71], resid_t)
    pp.close()
    print(f'Plotted the {res} spectral sequence')
    return


def standards(res: str):
    """Plots the standards as a multi page pdf

    Parameters
    ----------
    res : str
        The resolution of the standards to be plotted
    """
    pp = PdfPages(f'alt_redspec/alt_{res}_standards.pdf')
    c = 2
    plot_list = glob.glob(f'alt_redspec/standards/*{res}*txt')
    plot_list.sort()
    for i in plot_list:
        plt.figure(c, figsize=(20, 10))
        ax = plt.subplot(111)
        wave, flux, error = np.loadtxt(i, unpack=True)
        ob = i.split('/')[-1].split('_')[0]
        prog = i.split('/')[-1].split('_')[2]
        actualused = i.split('/')[-1].split('_')[-1].strip('.txt')
        name = i.split('/')[-1].split('_')[-2].strip('.txt')
        rc('text', usetex=False)
        plt.errorbar(wave, flux, yerr=error, label=f'{ob} {prog}')
        plt.title(f'{name.upper()} reduced using {actualused.upper()}')
        rc('text', usetex=True)
        plt.xlabel(r'Wavelength $\AA$')
        plt.ylabel(r'$F_{\lambda}$ $erg/cm^{2}/s/\AA$', color='blue')
        plt.yscale('log')
        plt.legend(fontsize='x-large')
        try:
            ax2 = ax.twinx()
            realname = f"calib_models/{name}_mod.txt"
            waves, fluxs = np.loadtxt(realname, unpack=True, usecols=(0, 1))
        except OSError:
            pass
        else:
            fluxs = fluxs[waves > wave.min()]
            waves = waves[waves > wave.min()]
            fluxs = fluxs[waves < wave.max()]
            waves = waves[waves < wave.max()]
            ax2.plot(waves, fluxs, color='red', label='Literature')
            plt.ylabel(r'$F_{\lambda}$ $erg/cm^{2}/s/\AA$', color='red')
            plt.yscale('log')
        pp.savefig()
        plt.close()
        c += 1
    pp.close()
    print(f'Plotted the {res} standards')
    return


def objects(res: str):
    """Plots the objects as a multi page pdf

    Parameters
    ----------
    res : str
        The resolution of the objects to be plotted
    """
    pp = PdfPages(f'alt_redspec/alt_{res}_objects.pdf')
    c = 1
    plot_list = glob.glob(f'alt_redspec/objects/*{res}*txt')
    plot_list.sort()
    for i in plot_list:
        plt.figure(c, figsize=(8, 6))
        wave, flux, error = np.loadtxt(i, unpack=True)
        wave = wave[flux > 0]
        error = error[flux > 0]
        flux = flux[flux > 0]
        ob = i.split('/')[-1].split('_')[0]
        prog = i.split('/')[-1].split('_')[-2]
        name = i.split('/')[-1].split('_')[-1].strip('.txt')
        rc('text', usetex=False)
        plt.errorbar(wave, flux * wave, yerr=error, label=f'{ob} {prog}')
        plt.title(name)
        rc('text', usetex=True)
        plt.xlabel(r'Wavelength $\AA$')
        plt.ylabel(r'Log $\lambda F_{\lambda}$ $erg/cm^{2}/s/\AA$')
        plt.legend()
        plt.yscale('log')
        pp.savefig()
        plt.close()
        c += 1
    pp.close()
    print(f'Plotted the {res} objects')
    return


def residuals(res: str, ttype: str):
    """Plots the residuals of extracted regions

    Parameters
    ----------
    res: str
        The resolution of the cut out regions to be plotted
    ttype: str
        The type of target cut out regions to be plotted (object or standard)
    """
    rc('text', usetex=False)
    pp = PdfPages(f'alt_redspec/alt_residuals_{res}_{ttype}.pdf')
    c = 1
    plot_list = glob.glob(f'alt_redspec/residuals/{ttype}/*{res}*txt')
    plot_list.sort()
    for i in plot_list:
        plt.figure(c)
        try:
            data = np.loadtxt(i)
        except ValueError:
            print(i)
            continue
        plt.imshow(data, origin='lower', cmap='RdBu', extent=(0, len(data[0]) * 20, 0, len(data)))
        name = i.split('/')[-1].split('_')[-1].strip('.txt')
        ob = i.split('/')[-1].split('_')[0]
        prog = i.split('/')[-1].split('_')[-2]
        plt.title(f'{name}_{ob}_{prog}')
        plt.yticks([])
        plt.xticks([])
        plt.colorbar()
        pp.savefig()
        plt.close()
        c += 1
    pp.close()
    print(f'Plotted the {res} {ttype} residuals')
    return


def calib_functions(res: str):
    """Plots the calibration function

    Parameters
    ----------
    res : str
        The resolution of the calibration functions to be plotted
    """
    pp = PdfPages(f'alt_redspec/alt_{res}_functions.pdf')
    c = 1
    plot_list = glob.glob(f'alt_redspec/calib_funcs/*{res}*txt')
    plot_list.sort()
    for i in plot_list:
        plt.figure(c, figsize=(20, 10))
        wave, func = np.loadtxt(i, unpack=True)
        plt.plot(wave, func, label=i.split('/')[-1].split('_')[0])
        rc('text', usetex=False)
        plt.title(i.split('/')[-1].split('_')[2] + ' to ' + i.split('/')[-1].split('_')[-1].strip('.txt'))
        rc('text', usetex=True)
        plt.xlabel(r'Wavelength $\AA$')
        plt.ylabel(r'Calibration Function')
        plt.legend()
        pp.savefig()
        plt.close()
        c += 1
    pp.close()
    print(f'Plotted the {res} calibration functions')
    return


def limited_sequence():
    """Plots one object per spectral type into a sequence using splat"""
    chosen_dict = {'M9': 'OB0005_R2500I_GTC8-15ITP_J0502+1442.txt',
                   # 'M9.5': 'OB0008_R2500I_GTC54-15A0_J1221+0257.txt',
                   # 'M9.5 beta': 'OB0005_R2500I_GTC54-15A0_J0953-1014.txt',
                   'L0': 'OB0019_R2500I_GTC8-15ITP_J1412+1633.txt',
                   # 'L0.5': 'OB0001_R2500I_GTC8-15ITP_J0028-1927.txt',
                   'L1': 'OB0015_R2500I_GTC8-15ITP_J1127+4705.txt',
                   'L2': 'OB0002_R2500I_GTC8-15ITP_J0235-0849.txt',
                   # 'L2.5': 'OB0018_R2500I_GTC8-15ITP_J1346+0842.txt',
                   'L3': 'OB0008_R2500I_GTC8-15ITP_J0823+6125.txt',
                   # 'L3 gamma': 'OB0024_R2500I_GTC54-15A0_J1004+5022.txt',
                   # 'L3.5': 'OB0031_R2500I_GTC8-15ITP_J2339+3507.txt',
                   'L4': 'OB0034_R2500I_GTC54-15A0_J1246+4027.txt',
                   # 'L4.5': 'OB0023_R2500I_GTC8-15ITP_J1539-0520.txt',
                   'L5': 'OB0030_R2500I_GTC54-15A0_J1213-0432.txt',
                   # 'L5.5': 'OB0021_R2500I_GTC54-15A0_J1750-0016.txt',
                   'L6': 'OB0027_R2500I_GTC8-15ITP_J1717+6526.txt'}
    # chosen_dict = {key: chosen_dict[key] for key in [*chosen_dict.keys(), ][::-1]}  # invert dictionary order
    splist, labels, c, zpoints = [], [], 1, []
    for k in chosen_dict:
        f = chosen_dict[k]
        sname = f.split('_')[-1].strip('.txt')
        f = 'alt_redspec/objects/' + f
        spt = k
        sp = Spectrum(filename=f, wunit=u.Angstrom, funit=(u.erg / u.cm ** 2 / u.Angstrom / u.s))
        sp.normalize(waverange=[8100, 8200])
        idx = np.logical_and(sp.flux.value < 3, sp.flux.value > 0)
        sp.wave = sp.wave[idx]
        sp.flux = sp.flux[idx] * c
        zpoints.append(c)
        c /= 100
        splist.append(sp)
        labels.append(f'{sname}: {spt}')
    plotSpectrum(splist, labels=labels, figsize=[8, 12], colorScheme='copper',
                 legendLocation='outside', features=['H2O', 'TiO', 'FeH'],
                 # zeropoint=np.logspace(0, -(len(chosen_dict) - 1), len(chosen_dict)),
                 xlabel=r'Wavelength [$\AA$]',
                 ylabel=r'Normalised $\mathrm{F_{\lambda}}$ [$\mathrm{erg}\ \AA^{-1}\ \mathrm{cm^{-2}}\ s^{-1}$]',
                 linestyle='solid', yrange=[c, 10], ylog=True, output='concise_spec_sequence.pdf')
    print('Plotted the concise spectral sequence')
    return


def main():
    """Main function

    This is the main method of the script, in which it controls all the different different plots, designed to be
    multi-threaded over the available cores.
    """
    # processes = []
    tnew300, tnew2500 = tabulate()
    objects('R0300R')
    objects('R2500I')
    # for i in ((tnew300, 'R0300R'), (tnew2500, 'R2500I')):
    #     p = multiprocessing.Process(target=spec_sequence, args=i)
    #     processes.append(p)
    #     p.start()
    # for i in ('R0300R', 'R2500I'):
    #     p = multiprocessing.Process(target=standards, args=(i, ))
    #     processes.append(p)
    #     p.start()
    #     p2 = multiprocessing.Process(target=objects, args=(i, ))
    #     processes.append(p2)
    #     p2.start()
    #     p3 = multiprocessing.Process(target=calib_functions, args=(i, ))
    #     processes.append(p3)
    #     p3.start()
    # for i in (('R0300R', 'objects'), ('R0300R', 'standards'), ('R2500I', 'objects'), ('R2500I', 'standards')):
    #     p = multiprocessing.Process(target=residuals, args=i)
    #     processes.append(p)
    #     p.start()
    # p = multiprocessing.Process(target=limited_sequence)
    # processes.append(p)
    # p.start()
    # for process in processes:
    #     process.join()
    # print('Finished processes.')
    # limited_sequence()
    return


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in ('--help', '-h'):  # if called command line 'python alt_plot.py --help
        print(__doc__)  # then print the documentation
    else:
        rcParams.update({'axes.labelsize': 'small', 'axes.titlesize': 'small',
                         'xtick.labelsize': 'small', 'ytick.labelsize': 'small',
                         'legend.fontsize': 'small', 'font.serif': ['Helvetica', 'Arial',
                                                                    'Tahoma', 'Lucida Grande',
                                                                    'DejaVu Sans'],
                         'font.family': 'serif', 'legend.frameon': False, 'legend.facecolor': 'none',
                         'mathtext.fontset': 'cm', 'mathtext.default': 'regular',
                         'figure.figsize': [4, 3], 'figure.dpi': 144, 'lines.linewidth': .75,
                         'xtick.top': True, 'ytick.right': True, 'legend.handletextpad': 0.5,
                         'xtick.minor.visible': True, 'ytick.minor.visible': True})
        main()
