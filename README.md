A Python-3 script to reduce GTC OSIRIS longslit optical spectra. Currently only supporting R300R and R2500I.
It is designed to be similar in function to IRAF/ apall.
There is also a script to plot the resultant data.
It is currently only useful for my data and file structure but should be full released in Cooper et al. (in prep)
<b> This is very much a WIP, there are an unacceptable number of hard coded variables and files.
Eventually these will be removed and the code edited to allow anyone to easily use.</b>

It is designed to work over multiple processors and will require a folder 'alt_redspec/' containing
'objects/', 'standards/', 'residuals/' and 'calib_funcs/'. On the upper level it will place the plots
(made in the alt_plot.py script) of the resultant spectra.

It will search for spectra in the current directory sub-folder 'Raw/' in which we expect in subsequent order:
'resolution/programme_ID/observing_block/' where those folders correspond with YOUR values
(e.g. 'Raw/R2500I/GTC54-15ITP/OB0001/'), inside each observing block directory we also expect:
'bias/', 'flat/', 'object/' and 'stds/' inside which are the fits files beginning with '0'.

