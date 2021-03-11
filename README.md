A Python-3 script to reduce GTC OSIRIS longslit optical spectra.

It is designed to work over multiple processors and will require a directory for the reduced spectra to be put in,
this is the redpath argument in the config file.

It will search for raw spectra in the config rawpath argument (without ending slash) in which we expect:
'resolution/programme_ID/observing_block/' inside each observing block directory we also expect:
'bias/', 'flat/', 'object/', 'arc/' and 'stds/' inside which are the fits files beginning with '0'.

**Config parameters include:**  
**rawpath** : str  
    Path to the raw spectra  
**redpath** : str  
    Path to the reduced spectra  
**targetlist** : str (optional)  
    Name of a file that contains an the header name and actual target name  
**head_actual** : str (optional)  
    The column names in said targetlistl, split as head_actual to convert from header name to actual target name  
**minpix** : int (pixel)  
    The minimum pixel on the dispersion axis to reduce within  
**maxpix** : int (pixel)  
    The maximum pixel on the dispersion axis to reduce within  
**stripewidth** : int (pixel)  
    The width in pixels over which to determine background/ find the source (larger=better but beware of shifts)  
**cpix** : int (pixel)  
    The central pixel one could typically find the spectra (not used in actual extraction)  
**minwave** : int (Angstroms)  
    The mimimum wavelength of the grism (used to cut the line list)  
**maxwave** : int (Angstroms)  
    The maximum wavelength of the grism (used to cut the line list)  
**maxthread** : int  
    The number of threads to use multiprocessing on  

**Required file:**  
    * <name>.config  -- file containing config arguments, see examples ```all_r2500.config```, ```all_r300.config```
