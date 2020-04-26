"""features.py

Python implementation of ModelImageWithSmallGaborSet
"""
import cv2, numpy, tqdm
from numpy import pi, ceil
import matplotlib.pyplot as plt
from os.path import join
import itertools

## settings
plotdir = 'plots'
img_fpath = 'images/dog_200.png'
n_sfs = 12
n_oris = 6

## other constants
bandwidth_constant = 0.56   ## 0.56 corresponds to bandwidth of 1
min_sf = 1.5                ## lowest spatial frequency in cyles / image
gamma = 0.5                 ## spatial aspect ratio a.k.a "ellipsicity"; 1 is round, 0 straight line
min_wavelength_pix = 4      ## smallest wavelength in pixels
kernel_extent = 4           ## how far to extend kernel from center in std of the gaussian (sigma)

## read image
image = cv2.imread(img_fpath)                           ## 3 channels, uint8
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255     ## 1 channel float

## derived settings
size = image.shape[0]
max_sf = size / min_wavelength_pix
spatial_freqs = numpy.geomspace(min_sf, max_sf, num=n_sfs)
orientations = numpy.arange(n_oris) * (pi/n_oris)
ori_freq_idx = list(itertools.product(range(n_oris), range(n_sfs)))


## loop with indices instead to be able to index gain

for o, f in tqdm.tqdm(ori_freq_idx, desc='filtering'):
    lambd = size / sf                   ##  wavelength of the sinusoidal factor.
    sigma = lambd * bandwidth_constant  ##  standard deviation of the gaussian envelope
    kside = 1 + 2 * int(ceil(kernel_extent * sigma))
    shape = (kside, kside)
    ktype = cv2.CV_32F                  ##  type of filter coefficients. It can be CV_32F or CV_64F .(float32)

    ## create kwargs dict here.
    kernel_real = cv2.getGaborKernel(shape, sigma, theta, lambd, gamma, psi=0, ktype=ktype)
    kernel_imag = cv2.getGaborKernel(shape, sigma, theta, lambd, gamma, psi=pi/2, ktype=ktype)
    filt_real = cv2.filter2D(image, -1, kernel_real)
    filt_imag = cv2.filter2D(image, -1, kernel_imag)
    filt_mag = numpy.abs(filt_real + 1j * filt_imag)

for o, f in tqdm.tqdm(ori_freq_idx, desc='local selection'):