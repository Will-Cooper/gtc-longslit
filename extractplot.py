import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import numpy as np

import argparse
import json

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
myargs.add_argument('-f', '--file', help='Relative path and .json file to plot', required=True, type=str)
myargs.add_argument('-s', '--low-signal', help='If low signal, select to decrease y range',
                    default=False, action='store_true')
myargs.add_argument('-k', '--save', help='Save file?', default=False, action='store_true')
args = myargs.parse_args()
fname = args.file
lowsig = args.low_signal
dosave = args.save


def animate(i: str):
    d = jdict[i]
    title.set_text(i)
    line1.set_offsets(np.array((d['xsmall'], d['region'])).T)
    line2.set_xdata(d['xbig'])
    line2.set_ydata(d['ybig'])
    line3.set_xdata(d['xbig'])
    line3.set_ydata(d['yfit'])
    y = [0, d['params'][3]]
    line4.set_xdata([d['params'][1], ] * 2)
    line4.set_ydata(y)
    line5.set_xdata([d['params'][0], ] * 2)
    line5.set_ydata(y)
    line6.set_xdata([d['params'][2], ] * 2)
    line6.set_ydata(y)
    line7.set_ydata([d['params'][3], ] * 2)
    line8.set_ydata([d['params'][5], ] * 2)
    line9.set_ydata([d['params'][4], ] * 2)
    return line1, line2, line3, line4, line5, line6, line7, title, line8, line9


def animinit():
    title.set_text('0')
    line1.set_offsets(np.array((xsmallinit, ysmallinit)).T)
    line2.set_xdata(xbiginit)
    line2.set_ydata(ybiginit)
    line3.set_xdata(xbiginit)
    line3.set_ydata(yfitinit)
    line4.set_xdata([cinit, ] * 2)
    line4.set_ydata(yran)
    line5.set_xdata([cminit, ] * 2)
    line5.set_ydata(yran)
    line6.set_xdata([cpinit, ] * 2)
    line6.set_ydata(yran)
    line7.set_xdata(xran)
    line7.set_ydata(amp)
    line8.set_xdata(xran)
    line8.set_ydata([back, ] * 2)
    line9.set_xdata(xran)
    line9.set_ydata([minreg, ] * 2)
    return line1, line2, line3, line4, line5, line6, line7, title, line8, line9


with open(fname, 'r') as jfile:
    jdict = json.load(jfile)
print('json loaded')
fig, ax = plt.subplots(figsize=(8, 5))
xsmallinit = jdict['0']['xsmall']
ysmallinit = jdict['0']['region']
xbiginit = jdict['0']['xbig']
ybiginit = jdict['0']['ybig']
yfitinit = jdict['0']['yfit']
cinit = jdict['0']['params'][1]
cminit = jdict['0']['params'][0]
cpinit = jdict['0']['params'][2]
amp = jdict['0']['params'][3]
minreg = jdict['0']['params'][4]
back = jdict['0']['params'][5]
xran, yran, y0 = [0, 100], [amp, amp], [0, amp]
line1 = ax.scatter(xsmallinit, ysmallinit, marker='s', s=4, color='black', label='Data')
line2, = ax.plot(xbiginit, ybiginit, color='blue', label='Interpolated')
line3, = ax.plot(xbiginit, yfitinit, color='orange', label='Gaussian Fit')
line4, = ax.plot([cinit, ] * 2, y0, color='grey', label='Center', lw=0.5)
line5, = ax.plot([cminit, ] * 2, y0, color='grey', label='Extraction limits', ls='--', lw=1)
line6, = ax.plot([cpinit, ] * 2, y0, color='grey', ls='--', lw=1)
line7, = ax.plot(xran, yran, color='grey', ls='--', lw=1, label='Amplitude')
line8, = ax.plot(xran, [back, ] * 2, color='black', ls='--', label='Background')
line9, = ax.plot(xran, [minreg, ] * 2, color='blue', ls='--', label='Minimum')
if lowsig:
    ax.set_ylim(0, 32535)
    title = ax.text(5, 3e4, '0')
else:
    ax.set_ylim(0, 65535)
    title = ax.text(5, 6e4, '0')
ax.set_xlim(0, 100)
ax.set_xlabel('Row')
ax.set_ylabel('Count')
ax.legend()
print('initialised')
ani = animation.FuncAnimation(fig, animate, frames=jdict, interval=16, init_func=animinit,
                              blit=True, save_count=len(jdict), repeat=False)
fnames = fname.split('/')
fnameout = '/'.join(fnames[:-1]) + '/vids/' + fnames[-1][:fnames[-1].find('.json')] + '.mp4'
print('running...')
if dosave:
    ani.save(fnameout, writer=animation.FFMpegWriter(fps=60))
else:
    plt.show()
