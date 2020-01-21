A Python-3 script to reduce GTC OSIRIS longslit optical spectra. Currently only supporting R300R and R2500I.

It is designed to work over multiple processors and will require a folder 'alt_redspec/' containing
'objects/', 'standards/', 'residuals/' and 'calib_funcs/'. On the upper level it will place the plots
(made in the alt_plot.py script) of the resultant spectra.

It requires packages: numpy, glob, matplotlib, astropy, scipy, warnings, sys, multiprocessing, time and typing.

It will search for spectra in the current directory sub-folder 'Raw/' in which we expect in subsequent order:
'resolution/programme_ID/observing_block/' where those folders correspond with YOUR values
(e.g. 'Raw/R2500I/GTC54-15ITP/OB0001/'), inside each observing block directory we also expect:
'bias/', 'flat/', 'object/' and 'stds/' inside which are the fits files beginning with '0'.

Required non script files/ folders in the same directory as this script include:
    * Master_info_correct_cm.csv    -- Containing the filenames, programme IDs, observing blocks, resolutions,
                                       shortnames, distances, spectral type and spectral type number.
    * observing_qc_positions_parseable.txt  -- Containing the programme IDs, observing bocks, resolutions
                                               and x pixel where the spectra is on the image (default is 250).
    * calib_models/     -- Containing the WD models that correspond with the observed standards
    * alt_done.log      -- An empty text file to be filled with the files that have been reduced already
    * good_wds.txt      -- A descriptive file of which standards per observing block should be used
    * BPM_(resolution)_python.txt       -- A file for the bad pixel masks for the resolution (else one will be made)
    * alt_doplot.log    -- A descriptive file for which spectra should be plotted per observing block
    * (resolution)_arcmap.txt       -- A file of pixel and wavelength
    * R300R_correction_function.txt     -- Correcting second order contamination in R300R spectra

Classes
-------
OB : str
    The full class, passed the string to the observing block and hence reduces the spectra in that observing block
BPM : str
    Determines the bad pixel mask if the corresponding resolution mask does not exist in the current environment

Methods
-------
repeat_check
    Checks if an object has been observed in multiple observing blocks and median stacks if so
main
    Main function of the script to make bad pixel mask and reduce the spectra

