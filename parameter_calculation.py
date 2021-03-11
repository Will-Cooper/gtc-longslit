"""
A script to calculate astrophysical parameters from spectra
"""
import glob
from typing import Tuple, Union
import multiprocessing
import argparse

import splat
import numpy as np
import scipy.interpolate as sinterp
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from bokeh.plotting import output_file, show, figure
from bokeh.layouts import column
from bokeh.models import Whisker, ColumnDataSource
from bokeh.io import export_png


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
        if self.spt == 0:
            return 0
        s = self.spt - 60
        teff = 4400.9 - 467.26 * s + 54.67 * s ** 2 - 4.4727 * s ** 3 + 0.17767 * s ** 4 - 0.0025492 * s ** 5
        try:
            return int(teff)
        except np.ma.core.MaskError:
            return 0

    def get_distance_spt(self) -> Tuple[float, float, str]:
        """Gets the name of the object"""
        t = Table.read('Master_info_correct_cm.csv')
        t['jdksptnum'].fill_value = 0
        t['truejdkspt'].fill_value = 'none'
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


def spt_from_num(sptnum: Union[float, int]) -> str:
    """Converts spectral type number to spectral type"""
    def spt_make(sptype: str, sub: int) -> str:
        """Joins strings"""
        def round_val() -> str:
            """Rounds values to nearest 0.5"""
            val = sptnum - sub
            if val >= 10:
                raise ArithmeticError('Spectral type number incorrect')
            rval = 0.5 * round(val / 0.5)
            return str(rval)
        return "".join([sptype, round_val()])
    if sptnum < 70:
        spt = spt_make('M', 60)  # M dwarf
    elif sptnum < 80:
        spt = spt_make('L', 70)  # L dwarf
    elif sptnum < 90:
        spt = spt_make('T', 80)  # T dwarf
    else:
        spt = spt_make('Y', 90)  # Y dwarf
    if '.0' in spt:
        return spt[:2]
    else:
        return spt


def plot_typing(all_objects: list) -> None:
    """Plots on a multipage pdf the different objects and their determined spectral types"""
    def normalise(arr: np.ndarray, idx: np.ndarray) -> float:
        """Finds the median value of the normalisation region"""
        norm_region = arr[idx]
        return np.nanmedian(norm_region)

    def get_norm(sptnumcheck: Union[float, int]) -> int:
        """Gets the normalisation constant"""
        if sptnumcheck < 70:
            return 7500
        else:
            return 8150

    def norm_idx(arr: np.ndarray) -> np.ndarray:
        """Get the wavelength indices to be normalised by"""
        return np.flatnonzero(np.logical_and(arr > norm - 50, arr < norm + 50))

    def log_trim(arrwave: np.ndarray, arrflux: np.ndarray, arrfluxerr: np.ndarray = None):
        """Trims arrays of tiny values after they've been normalised"""
        idx = np.flatnonzero(arrflux > 0.01)
        if arrfluxerr is None:
            return arrwave[idx], arrflux[idx]
        else:
            return arrwave[idx], arrflux[idx], arrfluxerr[idx]
    pp = PdfPages('normalisation_typing.pdf')
    t = Table.read('physical_parameters.csv', format='csv')
    c = 1
    for f in all_objects:
        wave, flux, fluxerror = Full.load_target(f)
        name = f.split('_')[-1].strip('.txt')
        res = f.split('/')[-1].split('_')[1]
        for row in t:
            if row['Name'] == name:
                davytype = row['SpT']
                mytype = row['MySpT']
                break
        else:
            davytype, mytype = '', ''
        mytypenum = num_from_spt(mytype)
        twavem, tfluxm = Full.davy_open(mytypenum)
        plt.figure(1)
        try:
            davytypenum = num_from_spt(davytype)
        except ValueError:
            davytypenum = 0
            norm = get_norm(mytypenum)
        else:
            twaved, tfluxd = Full.davy_open(davytypenum)
            norm = get_norm(davytypenum)
            twnormd = norm_idx(twaved)
            tfluxdnorm = normalise(tfluxd, twnormd)
            tfluxd /= tfluxdnorm
            twaved, tfluxd = log_trim(twaved, tfluxd)
            plt.plot(twaved, tfluxd, label=f'{davytype}: By Eye', color='blue')
        try:
            pwave, pflux = np.loadtxt(f'old/prev_reduc/{name}.txt', unpack=True, skiprows=1)
        except OSError:
            pass
        else:
            pwnorm = norm_idx(pwave)
            pfluxnorm = normalise(pflux, pwnorm)
            # plt.errorbar(pwave, pflux / pfluxnorm, yerr=pfluxerr / pfluxnorm, label=f'{name} Previous', color='green')
            pflux /= pfluxnorm
            pwave, pflux = log_trim(pwave, pflux)
            plt.plot(pwave, pflux, label=f'{name} Previous', color='green')
        wnorm = norm_idx(wave)
        twnormm = norm_idx(twavem)
        normflux = normalise(flux, wnorm)
        tfnormm = normalise(tfluxm, twnormm)
        flux /= normflux
        wave, flux, fluxerror = log_trim(wave, flux, fluxerror)
        tfluxm /= tfnormm
        twavem, tfluxm = log_trim(twavem, tfluxm)
        if mytypenum != davytypenum:
            plt.plot(twavem, tfluxm, label=f'{mytype}: Normalised Fit', color='orange')
        plt.errorbar(wave, flux, yerr=fluxerror, label=f'{name} - {res}', color='black')
        plt.legend()
        plt.ylabel(f'Flux Normalized at {norm} Angstroms')
        plt.xlabel('Wavelength (Angstroms)')
        plt.title(f'{name}')
        plt.yscale('log')
        pp.savefig(bbox_inches='tight')
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


def indices_plotting(indices_list: list):
    """Plots the spectral indices calculated"""
    def source_split(r: str) -> ColumnDataSource:
        res_bool = reses == r
        xvals = sptypes[res_bool]
        yvals = ind_vals[res_bool]
        nvals = names[res_bool]
        res_arr = reses[res_bool]
        errvals = ind_valerr[res_bool]
        uptop = upper[res_bool]
        botlow = lower[res_bool]
        return ColumnDataSource(data=dict(x=xvals, y=yvals, obj_name=nvals, resolution=res_arr, err=errvals,
                                          upper=uptop, lower=botlow))
    if len(indices_list) == 0:
        raise FileNotFoundError('Appears to be empty list')
    tdata = Table.read('physical_parameters.csv', format='csv')
    inds = ["Rb-a", "Rb-b", "Na-a", "Na-b", "Cs-a", "Cs-b", "TiO-a", "TiO-b", "VO-a", "VO-b", "CrH-a", "CrH-b", "FeH-a",
            "FeH-b", "Color-a", "Color-b", "Color-c", "Color-d", "PC3", "PC6", "CrH1", "CrH2", "FeH1", "FeH2", "H2O1",
            "TiO1", "TiO2", "VO1", "VO2"]
    numinds = len(inds)
    indserr = ["".join([i, '_err']) for i in inds]
    output_file('indices_plotted.html', title='Spectral Indices')
    tplot = Table(names=['name', 'sptnum', 'resolution'] + inds + indserr)
    tplot['name'].dtype = 'U12'
    tplot['resolution'].dtype = 'U8'
    for f in indices_list:
        t = Table.read(f, format='csv')
        name = f.split('/')[-1].split('_')[0]
        for row in tdata:
            if name == row['Name']:
                try:
                    sptnum = num_from_spt(row['SpT'])
                except ValueError:
                    sptnum = num_from_spt(row['MySpT'])
                res = row['Res']
                break
        else:
            sptnum = 0
            res = ''
        if sptnum == 0:
            continue
        row = np.empty_like(tplot.colnames)
        for i, col in enumerate(tplot.colnames):
            if 'err' in col:
                break
            elif i == 0:
                row[i] = name
            elif i == 1:
                row[i] = sptnum
            elif i == 2:
                row[i] = res
            else:
                try:
                    val, err = t[col]
                except KeyError:
                    row[i] = np.nan
                    row[i + numinds] = np.nan
                else:
                    row[i] = float(f'{val:.4f}')
                    row[i + numinds] = float(f'{err:.4f}')
        tplot.add_row(row)
    sptypes = tplot['sptnum']
    minspt, maxspt = np.floor(np.min(sptypes)), np.ceil(np.max(sptypes))
    xlims = np.linspace(minspt, maxspt, int(maxspt - minspt + 1), dtype=int)
    names = np.array(tplot['name'])
    reses = np.array(tplot['resolution'])
    glyphs, sptdict = [], {}
    tooltips = [('Name', '@obj_name'), ('Spectral Index', '@y +/- @err'), ('Resolution', '@resolution')]  # hover tool
    for i in xlims:
        sptdict[int(i)] = spt_from_num(i)
    for i in inds:
        ind_vals = np.array(tplot[i])
        if np.all(np.isnan(ind_vals)):
            continue
        ind_valerr = np.array(tplot["".join([i, '_err'])])
        upper = ind_vals + ind_valerr
        lower = ind_vals - ind_valerr
        pfit = np.polyfit(sptypes[np.logical_not(np.isnan(ind_vals))], ind_vals[np.logical_not(np.isnan(ind_vals))], 4)
        p1d = np.poly1d(pfit)
        p = figure(tools="pan,hover,box_zoom,reset,wheel_zoom,save", tooltips=tooltips, active_drag='box_zoom',
                   title=i, x_axis_label='Spectral Type', y_axis_label='Spectral Index',
                   plot_height=366, plot_width=488)
        sourcer25 = source_split('R2500I')
        p.circle(source=sourcer25, x='x', y='y', color='blue', size=8,
                 legend_label='R2500I', muted_color='blue', muted_alpha=0.1)
        p.add_layout(Whisker(source=sourcer25, base="y", upper="upper", lower="lower",
                             dimension='height', level='overlay', line_color='blue'))
        sourcer03 = source_split('R0300R')
        p.square(source=sourcer03, x='x', y='y', color='orange', size=6,
                 alpha=0.75, legend_label='R300R', muted_color='orange', muted_alpha=0.1)
        p.add_layout(Whisker(source=sourcer03, base="y", upper="upper", lower="lower",
                             dimension='height', level='overlay', line_color='orange'))
        xfit = np.linspace(minspt, maxspt, 100)
        p.line(xfit, p1d(xfit), color='black', line_dash='dashed', alpha=0.5)
        p.xaxis.ticker = xlims
        p.xaxis.major_label_overrides = sptdict
        p.hover.mode = 'mouse'
        p.legend.click_policy = "mute"
        p.sizing_mode = 'stretch_width'
        p.title.align = 'center'
        p.title.text_font_size = '18pt'
        p.title.text_font_style = 'bold'
        p.min_border_left = 60
        p.min_border_right = 60
        p.xaxis.axis_label_text_font_size = '16pt'
        p.yaxis.axis_label_text_font_size = '16pt'
        p.xaxis.major_label_text_font_size = '16pt'
        p.yaxis.major_label_text_font_size = '16pt'
        p.legend.label_text_font_size = '16pt'
        glyphs.append(p)
    tplot.write('plotted_indices.csv', overwrite=True)
    if use_png:
        export_png(column(glyphs, sizing_mode='scale_width'))
    else:
        show(column(glyphs, sizing_mode='scale_width'))
    return


def main():
    """Main module"""
    all_objects = glob.glob('alt_redspec/objects/*txt')
    all_objects = object_list_sort(all_objects)
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
    if do_plot:
        plot_typing(all_objects)
        indices_plotting(glob.glob('spectral_indices/*csv'))
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
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    myargs.add_argument('-a', '--astrophysical-parameters', action='store_true', default=False, help='Calculate params')
    myargs.add_argument('-p', '--create-plots', action='store_true', default=False, help='Make plots')
    myargs.add_argument('-i', '--indices-only', action='store_true', default=False, help='Only plot indices')
    myargs.add_argument('-png', '--export-png', action='store_true', default=False, help='Export indices as png')
    args = myargs.parse_args()
    do_parameters = args.astrophysical_parameters
    do_plot = args.create_plots
    only_indices_plotting = args.indices_only
    use_png = args.export_png
    if only_indices_plotting:
        indices_plotting(glob.glob('spectral_indices/*csv'))
    else:
        main()
