"""This is a potentially interactive python script which quickly opens and plots spectra.
When called from command line, needs to be given a relative (or full) path to a file.
Additionally, typing anything after the path to file will force the script to attempt normalise at 8150 Angstroms.
Requires packages: sys, numpy, matplotlib. Written in Python 3 but should work retroactively."""
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import rc


def main():
    """Main module"""
    try:
        assert (sys.argv.__len__() >= 2), 'Require a file name to be given'
    except AssertionError as e:
        print(f'{e}, you can also type anything after the file name to produce a normalised plot')
        f = input('Type path to file here: ')
    else:
        f = sys.argv[1]  # file name

    try:
        wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1))  # load file
    except (OSError, FileNotFoundError) as e:
        print(e, 'Cannot find given file in: ', f)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fname = f.split('/')[-1][:-4].split('_')[-1]
        ax1.set_title(f'{fname} Linear')
        ax2.set_title(f'{fname} Log')
        rc('text', usetex=True)  # using latex
        ax1.set_xlabel(r'Wavelength $\AA$')

        if sys.argv.__len__() > 2:
            try:
                w8000 = np.argwhere(wave > 8150)[0][0]
            except IndexError:
                print('Cannot normalise to 8150 Angstroms')
                fluxnorm = 1
                ylab = r'$\lambda F_{\lambda}$'
            else:
                fluxnorm = flux[w8000]
                ylab = r'$\lambda F_{\lambda}$ Normalized to $8150\AA$'
        else:
            fluxnorm = 1
            ylab = r'$\lambda F_{\lambda}$'

        ax1.plot(wave, wave * flux / fluxnorm)
        ax1.set_ylabel(ylab)
        ax2.plot(wave, wave * flux / fluxnorm)
        ax2.set_yscale('log')
        ax2.set_xlabel(r'Wavelength $\AA$')
        ax2.set_ylabel(ylab)
        plt.show()
    return


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in ('--help', '-h'):
        print(__doc__)
    else:
        main()
