import splat
import numpy as np
import glob
from typing import Tuple
import scipy.interpolate as sinterp
from astropy.table import Table
import multiprocessing
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import astropy.units as u


class Full:

    def __init__(self, f: str) -> None:
        """Constructor method"""
        self.wave, self.flux, self.fluxerror = self.load_target(f)   # the file information
        self.name = f.split('_')[-1].strip('.txt')
        self.resolution = f.split('_')[2]
        self.distance, self.spt, self.sptname = self.get_distance_spt()
        self.expteff = self.expected_teff()
        self.myspt = self.davy_typing()
        print(f'Done with {self.name} ({self.sptname}) normalisation spectral typing as {self.myspt}.')
        self.teff2, self.logg2 = self.normalised_fit()
        self.lbol2 = self.infer_lbol(1, self.teff2)
        print(f'Done with {self.name} physical parameter fitting.')
        self.writing()
        print('Measuring spectral indices')
        self.find_indices(f)
        return

    @staticmethod
    def infer_lbol(rad: float, teff: float) -> str:
        """Uses Stefan-Boltzmann to get Lbol from radius and Teff"""
        solrad = 10  # Radius of sun in Jupiter radii
        solteff = 5775  # teff of sun
        lbol = (rad / solrad) ** 2 * (teff / solteff) ** 4
        return format(lbol, '0.2e')  # luminosity in solar units

    def expected_teff(self) -> int:
        """Determines the expected teff from the spectral type using Stephens et al. 2009"""
        s = self.spt - 60
        teff = 4400.9 - 467.26 * s + 54.67 * s ** 2 - 4.4727 * s ** 3 + 0.17767 * s ** 4 - 0.0025492 * s ** 5
        return int(teff)

    def get_distance_spt(self) -> Tuple[float, float, str]:
        """Gets the name of the object"""
        t = Table.read('Master_info_correct_cm.csv')
        for row in t:
            if row['SHORTNAME'].strip() == self.name:
                distance = row['dist']
                spt = row['jdksptnum']
                sptname = row['truejdkspt']
                break
        else:
            print(f'Cannot find distance for {self.name}, assuming L0 at 10pc.')
            distance = 10
            spt = 70
            sptname = 'L0'
        return distance, spt, sptname

    @staticmethod
    def load_target(f: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uses numpy to load a file and pull out wave, flux and flux error"""
        return np.loadtxt(f, unpack=True)

    @staticmethod
    def load_smallgrid(f: str) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """Loads the smaller BT-Settl grids used to fit (not full SED)"""
        wave, flux = np.loadtxt(f, unpack=True)
        f = f.split('/')[-1][4:].split('-')
        teff = int(100 * float(f[0]))
        logg = float(f[1][:-4])
        return teff, logg, wave, flux

    @staticmethod
    def chi2(expected: np.ndarray, observed: np.ndarray) -> float:
        top = (observed - expected) ** 2
        bottom = expected
        return np.sum(top / bottom)

    @staticmethod
    def region_select(wave: np.ndarray, flux: np.ndarray,
                      temp_wave: np.ndarray, temp_flux: np.ndarray,
                      minwave: int, maxwave: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        midwave = (maxwave - minwave) / 2. + minwave
        flux = flux[np.logical_and(wave >= minwave, wave < maxwave)]
        wave = wave[np.logical_and(wave >= minwave, wave < maxwave)]
        temp_flux = temp_flux[np.logical_and(temp_wave >= minwave, temp_wave < maxwave)]
        temp_wave = temp_wave[np.logical_and(temp_wave >= minwave, temp_wave < maxwave)]
        return wave, flux, temp_wave, temp_flux, midwave

    def grid_comp(self, temp_wave: np.ndarray, temp_flux: np.ndarray, minwave: int, maxwave: int) -> float:
        """Compares the wavelength and flux to that of the grid and returns chi2"""
        flux = self.flux[np.logical_and(self.wave >= np.min(temp_wave), self.wave <= np.max(temp_wave))]
        wave = self.wave[np.logical_and(self.wave >= np.min(temp_wave), self.wave <= np.max(temp_wave))]
        temp_flux = temp_flux[np.logical_and(temp_wave >= np.min(wave), temp_wave <= np.max(wave))]
        temp_wave = temp_wave[np.logical_and(temp_wave >= np.min(wave), temp_wave <= np.max(wave))]
        wave, flux, temp_wave, temp_flux, midwave = self.region_select(wave, flux,
                                                                       temp_wave, temp_flux,
                                                                       minwave, maxwave)
        if len(temp_wave) == 0 or len(wave) == 0:
            return np.inf
        if midwave > np.max(temp_wave) or midwave > np.max(wave):
            return np.inf
        f = sinterp.interp1d(temp_wave, temp_flux)
        flux = flux[np.logical_and(wave >= np.min(temp_wave), wave <= np.max(temp_wave))]
        wave = wave[np.logical_and(wave >= np.min(temp_wave), wave <= np.max(temp_wave))]
        temp_flux = f(wave)
        try:
            flux /= flux[np.argwhere(wave > midwave)[0][0]]
            temp_flux /= temp_flux[np.argwhere(temp_wave > midwave)[0][0]]
        except IndexError:
            return np.inf
        chi2 = self.chi2(temp_flux, flux)
        return chi2

    @staticmethod
    def davy_open(sptnum: int) -> Tuple[np.ndarray, np.ndarray]:
        with fits.open(f'templates/{sptnum}.fits') as stand:
            data = stand[0].data
            means = np.array(data[0])
            head = stand[0].header
            dlen = head['NAXIS1']
            wmin = head['CRVAL1']
            if 'CDELT1' in head.keys():
                wmax = wmin + dlen * head['CDELT1']
            else:
                wmax = wmin + dlen * head['CD1_1']
            if head['CTYPE1'].strip() == 'LOG':
                wave = np.logspace(wmin, wmax, dlen)
            else:
                wave = np.linspace(wmin, wmax, dlen)
        return wave, means

    def davy_test(self, twave: np.ndarray, tflux: np.ndarray, minwave: int, maxwave: int) -> float:
        tflux = tflux[twave >= np.min(self.wave)]
        twave = twave[twave >= np.min(self.wave)]
        flux = self.flux[np.logical_and(self.wave >= twave.min(), self.wave <= twave.max())]
        wave = self.wave[np.logical_and(self.wave >= twave.min(), self.wave <= twave.max())]
        wave, flux, temp_wave, temp_flux, midwave = self.region_select(wave, flux,
                                                                       twave, tflux,
                                                                       minwave, maxwave)
        if len(temp_wave) == 0 or len(wave) == 0 or midwave > np.max(temp_wave) or midwave > np.max(wave):
            return np.inf
        f = sinterp.interp1d(twave, tflux)
        tnorm = tflux[np.argwhere(twave > midwave)[0][0]]
        tflux = f(wave)
        norm = flux[np.argwhere(wave > midwave)[0][0]]
        return self.chi2(flux / norm, tflux / tnorm)

    def davy_typing(self) -> str:
        """Uses standards and select regions to spectrally type the objects"""
        conv_dict = {}
        t = Table(dtype=('U2', float), names=('SpT', 'chi2'))
        for i in np.arange(66, 79, dtype=int):
            if i < 70:
                conv_dict[i] = f'M{i - 60}'
            else:
                conv_dict[i] = f'L{i - 70}'
            twave, tflux = self.davy_open(i)
            t.add_row([conv_dict[i], self.region_match(twave, tflux, True)])
        minchi = np.argmin(t['chi2'])
        return t['SpT'][minchi]

    def region_match(self, temp_wave: np.ndarray, temp_flux: np.ndarray, spectraltype: bool = False) -> float:
        regions = np.array([[7450, 7550],
                            [7575, 7600],
                            [7600, 7625],
                            [7650, 7800],
                            [7800, 8000],
                            [8000, 8175],
                            [8160, 8230],
                            [8250, 8300],
                            [8300, 8430],
                            [8450, 8800],
                            [8850, 8900],
                            [8900, 9200],
                            [9300, 9875],
                            [9875, 9925],
                            [9950, 10100]], dtype=int)
        if spectraltype:
            all_chi = np.stack([self.davy_test(temp_wave, temp_flux, j[0], j[1]) for j in regions])
        else:
            all_chi = np.stack([self.grid_comp(temp_wave, temp_flux, i[0], i[1]) for i in regions])
        #  w = np.array([1 / len(i) for i in regions])
        w = np.array([1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2])
        w = w[np.isfinite(all_chi)]
        all_chi = all_chi[np.isfinite(all_chi)]
        try:
            waverage = np.average(all_chi, weights=w)
        except ZeroDivisionError:
            return np.inf
        return waverage

    def normalised_fit(self) -> Tuple[int, float]:
        """Normalised the model and object at 8150A and does chi^2 fit"""
        all_grids = glob.glob('bt_spectra/useful/*txt')
        all_grids = np.stack([self.load_smallgrid(gridf) for gridf in all_grids])
        all_chi = np.stack([(n[0], n[1], self.region_match(n[2], n[3])) for n in all_grids])
        min_chi = np.argmin(all_chi[:, 2])
        teff = int(all_chi[min_chi, 0])
        logg = all_chi[min_chi, 1]
        return teff, logg

    def find_indices(self, f: str):
        """Calculates the spectral indices using splat"""
        sp = splat.Spectrum(f, wunit=u.Angstrom)
        try:
            kind = splat.measureIndexSet(sp, ref='kirkpatrick')
        except IndexError:
            kind = []
        t = Table(data=kind)
        mind = splat.measureIndexSet(sp, ref='martin')
        t2 = Table(data=mind)
        t.add_columns(t2.columns.values())
        t.write(f'spectral_indices/{self.name}_indices.csv', format='csv', overwrite=True)
        return

    def writing(self):
        """Writes the results to a file"""
        with open('physical_parameters.csv', 'a+') as f:
            f.write(f'{self.name},{self.resolution},{self.sptname},{self.myspt},{self.spt},{self.expteff},'
                    f'{self.teff2},{self.logg2},{self.lbol2}\n')
        print(f'Done with {self.name}.')
        return


def num_from_spt(spt: str) -> int:
    """Converts spectral type to spectral type number"""
    if spt[0] == 'M':
        num = 60
    else:
        num = 70
    if len(spt) > 2:
        num += int(spt[1:2])
    else:
        num += int(spt[1:])
    return num


def plot_typing(all_objects: list) -> None:
    """Plots on a multipage pdf the different objects and their determined spectral types"""
    pp = PdfPages('normalisation_typing.pdf')
    t = Table.read('physical_parameters.csv', format='csv')
    c = 1
    for f in all_objects:
        wave, flux, fluxerror = Full.load_target(f)
        name = f.split('_')[-1].strip('.txt')
        for row in t:
            if row['Name'] == name:
                davytype = row['SpT']
                mytype = row['MySpT']
                break
        else:
            davytype, mytype = '', ''
        davytypenum = num_from_spt(davytype)
        mytypenum = num_from_spt(mytype)
        twaved, tfluxd = Full.davy_open(davytypenum)
        twavem, tfluxm = Full.davy_open(mytypenum)
        if davytypenum < 70:
            norm = 7500
        else:
            norm = 8150
        plt.figure(1, figsize=(15, 10))
        wnorm = np.argwhere(wave > norm)[0][0]
        twnormd = np.argwhere(twaved > norm)[0][0]
        twnormm = np.argwhere(twavem > norm)[0][0]
        normflux = flux / flux[wnorm]
        wave = wave[np.isfinite(normflux)]
        fluxerror = fluxerror[np.isfinite(normflux)]
        normflux = normflux[np.isfinite(normflux)]
        plt.plot(twaved, tfluxd / tfluxd[twnormd], label=f'{davytype}: By Eye', color='blue')
        if mytypenum != davytypenum:
            plt.plot(twavem, tfluxm / tfluxm[twnormm], label=f'{mytype}: Normalized Fit', color='orange')
        plt.errorbar(wave, normflux, yerr=fluxerror / flux[wnorm], label=f'{name}', color='black')
        plt.legend()
        plt.ylim([-0.5, 8])
        plt.ylabel(f'Flux Normalized at {norm} Angstroms')
        plt.xlabel('Wavelength (Angstroms)')
        plt.title(f'{name}')
        pp.savefig()
        plt.close()
        c += 1
    pp.close()
    return


def object_list_sort(all_objects: list) -> list:
    """Sort the object list by target name"""
    all_obs = np.stack([f.split('_J') for f in all_objects])
    t = Table(data=all_obs)
    t.sort('col1')
    out_objects = []
    for row in t:
        out_objects.append(row[0] + '_J' + row[1])
    return out_objects


def indices_plotting(indices_list: list) -> None:
    """Plots the spectral indices calculated"""
    tdata = Table.read('physical_parameters.csv', format='csv')
    inds = ["Rb-a", "Rb-b", "Na-a", "Na-b", "Cs-a", "Cs-b", "TiO-a", "TiO-b", "VO-a", "VO-b", "CrH-a", "CrH-b", "FeH-a",
            "FeH-b", "Color-a", "Color-b", "Color-c", "Color-d", "PC3", "PC6", "CrH1", "CrH2", "FeH1", "FeH2", "H2O1",
            "TiO1", "TiO2", "VO1", "VO2"]
    for f in indices_list:
        t = Table.read(f, format='csv')
        name = f.split('/')[-1].split('_')[0]
        for row in tdata:
            if name == row['Name']:
                sptnum = row['sptnum']
                break
        else:
            sptnum = 0
        # TODO: Finish this
    return


def main():
    """Main module"""
    all_objects = glob.glob('alt_redspec/objects/*txt')
    all_objects = object_list_sort(all_objects)

    do_parameters = True
    if do_parameters:
        with open('physical_parameters.csv', 'w+') as f:
            f.write('Name,Res,SpT,MySpT,sptnum,ExpectTeff,'
                    'Teff2,Logg2,Lbol2\n')
        avail_cores = multiprocessing.cpu_count() - 1 or 1  # available cores to thread over
        print(f'Threading over {avail_cores} cores.')
        pool = multiprocessing.Pool(processes=avail_cores)
        pool.map(Full, all_objects)
        pool.close()
        t = Table.read('physical_parameters.csv', format='csv')
        t.sort('Name')
        t.write('physical_parameters.csv', format='csv', overwrite=True)

    do_plot = True
    if do_plot:
        plot_typing(all_objects)
    return


if __name__ == '__main__':
    main()
