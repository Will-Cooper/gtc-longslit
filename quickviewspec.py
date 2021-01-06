"""This is a potentially interactive python script which quickly opens and plots spectra.
When called from command line, needs to be given a relative (or full) path to a file.
Additionally, typing anything after the path to file will force the script to attempt normalise at 8150 Angstroms.
Requires packages: numpy, matplotlib. Written in Python 3 but should work retroactively."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from matplotlib import rc
from astropy.io import fits


def davy_open(sptnum):
    """Load and normalise the template

    Parameters
    ----------
    sptnum : int
        The spectral type number of the template to be plotted

    Returns
    -------
    wave : numpy.ndarray
        The wavelength array
    flux : numpy.ndarray
        The normalised flux array
    """
    with fits.open('templates/' + str(sptnum) + '.fits') as temp:
        data = temp[0].data  # the template data
        means = np.array(data[0])   # just the means column
        head = temp[0].header  # the header
        dlen = head['NAXIS1']  # axis length
        wmin = head['CRVAL1']  # minimum wavele`ngth value
        if 'CDELT1' in head.keys():  # wavelength spacing either CDELT1 or CD1_1
            wmax = wmin + dlen * head['CDELT1']  # end wavelength
        else:
            wmax = wmin + dlen * head['CD1_1']  # end wavelength point
        if head['CTYPE1'].strip() == 'LOG':   # if log spaced
            wave = np.logspace(wmin, wmax, dlen)  # wavelength array
        else:
            wave = np.linspace(wmin, wmax, dlen)   # wavelength array
        norm = np.median(means[-10:])  # normalisation
    return wave, means / norm


def main():
    """Main module"""
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    myargs.add_argument('-n', '--normalise', type=int, default=False, help='Normalise Point in Angstroms')
    myargs.add_argument('-f', '--files', required=True, nargs='+', help='Files to be plotted')
    myargs.add_argument('-t', '--title', help='Title')
    myargs.add_argument('-l', '--legend', nargs='+', help='Legend Labels')
    myargs.add_argument('-s', '--spectral-type', help='Plot template of spectral type number')
    args = myargs.parse_args()
    normpoint = args.normalise
    flist = args.files
    title = args.title
    labs = args.legend
    spt = args.spectral_type
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), dpi=244)
    if not title:
        fname = 'Spectra'
    else:
        fname = title
    ax1.set_title(f'{fname} Linear')
    ax2.set_title(f'{fname} Log')
    rc('text', usetex=True)  # using latex
    minwave = np.inf
    for c, f in enumerate(flist):
        try:
            wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1))  # load file
        except (OSError, FileNotFoundError) as e:
            print(e, 'Cannot find given file in: ', f)
            continue
        except ValueError:
            wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1), skiprows=1)  # load file
        if normpoint:
            if normpoint > 0:
                try:
                    w8000 = np.flatnonzero(wave > normpoint)[0]
                except IndexError:
                    print('Cannot normalise')
                    fluxnorm = 1
                else:
                    fluxnorm = np.nanmedian(flux[w8000 - 10: w8000 + 9])
            else:
                normpointstr = str(normpoint)[1:]
                if normpointstr[c] == '1':
                    fluxnorm = np.median(flux[-10:])
                elif normpointstr[c] == '2':
                    fluxnorm = np.median(flux[:9])
                else:
                    fluxnorm = 1
        else:
            fluxnorm = 1

        if np.max(wave) < 1000:
            wave *= 10000
        ax1.plot(wave, wave * flux / fluxnorm, label=labs[c])
        ax2.plot(wave, wave * flux / fluxnorm, label=labs[c])
        if np.min(wave) < minwave:
            minwave = np.min(wave)
    if spt:
        wave, means = davy_open(spt)
        wavecut = np.flatnonzero(wave > minwave)[0]
        wave = wave[wavecut:]
        means = means[wavecut:]
        ax1.plot(wave, wave * means, label='Template')
        ax2.plot(wave, wave * means, label='Template')
    # ax1.set_xlabel(r'Wavelength $\AA$')
    ax1.set_ylabel(r'$\lambda F_{\lambda}$')
    ax2.set_xlabel(r'Wavelength $\AA$')
    ax2.set_ylabel(r'$\lambda F_{\lambda}$')
    ax2.set_yscale('log')
    ax1.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
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
                     'xtick.top': True, 'ytick.right': True, 'legend.handletextpad': 0.5,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True})
    main()
