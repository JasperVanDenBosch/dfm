"""features.py

Python implementation of ModelImageWithSmallGaborSet

"""
# pylint: disable=no-member
import cv2, numpy, tqdm, scipy.spatial, pandas
from scipy.stats import rankdata
from numpy import pi, ceil
import matplotlib.pyplot as plt
from os.path import join
import itertools
from dfm.reframe import reframe

## settings
plotdir = 'plots'
img_fpath = 'images/dog_200.png'
# img_fpath = 'images/beach_400-200.png'
n_sfs = 12
n_oris = 6

## other constants
bandwidth_constant = 0.56   ## 0.56 corresponds to bandwidth of 1
min_sf = 1.5                ## lowest spatial frequency in cyles / image
gamma = 0.5                 ## spatial aspect ratio a.k.a "ellipsicity"; 1 is round, 0 straight line
min_wavelength_pix = 4      ## smallest wavelength in pixels
kernel_extent = 4           ## how far to extend kernel from center in std of the gaussian (sigma)
pix_fraction = 0.15         ## fraction of pixels to be selected as location for feature with a given kernel
min_dist_between = 0.3      ## features closer than this many wavelengths are discarded

## read image
image = cv2.imread(img_fpath)                           ## 3 channels, uint8
image = 1/255* cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  ## 1 channel float

## derived settings
size = image.shape[0]
max_sf = size / min_wavelength_pix
frequencies = numpy.geomspace(min_sf, max_sf, num=n_sfs)
orientations = numpy.arange(n_oris) * (pi/n_oris)
ori_freq_idx = list(itertools.product(range(n_oris), range(n_sfs)))

gain = numpy.full([n_oris, n_sfs, size, size], numpy.nan)
for o, f in tqdm.tqdm(ori_freq_idx, desc='kernel gain'):
    wavelength = size / frequencies[f]
    gaussian_std = wavelength * bandwidth_constant
    kside = 1 + 2 * int(ceil(kernel_extent * gaussian_std))
    kernel_params = dict(
        ksize=(kside, kside),
        sigma=gaussian_std,
        theta=orientations[o],
        lambd=wavelength,
        gamma=gamma,
        ktype=cv2.CV_32F
    )
    kernel_real = cv2.getGaborKernel(psi=0, **kernel_params)
    kernel_imag = cv2.getGaborKernel(psi=pi/2, **kernel_params)
    filt_real = cv2.filter2D(image, -1, kernel_real)
    filt_imag = cv2.filter2D(image, -1, kernel_imag)
    gain[o, f, :, :] = numpy.abs(filt_real + 1j * filt_imag)

features = []
for o, f in tqdm.tqdm(ori_freq_idx, desc='local selection'):
    wavelength = size / frequencies[f]
    gaussian_std = wavelength * bandwidth_constant

    ## a) pixels above threshold for this kernel
    kernel_pix_ranks = rankdata(gain[o, f, :, :]).reshape([size, size])
    pix_rank_cutoff = (1 - pix_fraction) * (size ** 2)
    kernel_top_pix = kernel_pix_ranks > pix_rank_cutoff
    best_ori = gain[:, f, :, :].argmax(axis=0)
    Y, X = numpy.where(kernel_top_pix & (best_ori==o))
    
    ## b) local maxima
    max_iterations = kernel_top_pix.sum()
    remaining = numpy.full_like(X, True, dtype=bool) #  remaining
    for _ in range(max_iterations):
        if ~remaining.any():
            break
        new_peak_index = gain[o, f, Y[remaining], X[remaining]].argmax()
        x, y = (X[remaining][new_peak_index], Y[remaining][new_peak_index])
        features.append(dict(o=o, f=f, theta=orientations[o], lambd=wavelength, sigma=gaussian_std, x=x, y=y))
        remaining_coords = numpy.array([X[remaining], Y[remaining]]).T
        dists = scipy.spatial.distance.cdist(
            numpy.atleast_2d([x, y]),
            remaining_coords
        )
        too_close = numpy.squeeze(dists) <= (wavelength * min_dist_between)
        remaining[remaining] = ~too_close
    else:
        raise ValueError('No convergence when finding local maxima')

features = pandas.DataFrame(features)
gabor_vects = numpy.full([features.shape[0], size**2], numpy.nan)
for feature in tqdm.tqdm(features.itertuples(), desc='construct features'):
    f = feature.Index
    kside = 1 + 2 * int(ceil(kernel_extent * feature.sigma))
    kernel_params = dict(
        ksize=(kside, kside),
        sigma=feature.sigma,
        theta=feature.theta,
        lambd=feature.lambd,
        gamma=gamma,
        ktype=cv2.CV_32F
    )
    kernel = cv2.getGaborKernel(psi=0, **kernel_params)
    gabor = reframe(kernel, width=size, height=size, x=feature.x, y=feature.y)
    gabor_vects[f, :] = gabor.flatten()

print('determine explained variance by feature..')
covs = gabor_vects @ image.flatten()
areas = numpy.sum(numpy.abs(gabor_vects), axis=1)
expl_var = covs / areas
expl_var_ranks = numpy.flip(numpy.argsort(expl_var))
features['expl_var_rank'] = expl_var_ranks

## selection
selection = numpy.full_like(expl_var_ranks, False, dtype=bool)
img_vect = image.ravel()
selection[expl_var_ranks[0]] = True   ## select first gabor
recon_vect = gabor_vects[selection, :].sum(axis=0)
recon_r = numpy.corrcoef(recon_vect, img_vect)[0, 1]
for i in tqdm.tqdm(range(1, expl_var_ranks.size), desc='selecting ranked gabors'):
    new_recon_vect = recon_vect + gabor_vects[expl_var_ranks[i], :]
    new_r = numpy.corrcoef(new_recon_vect, img_vect)[0, 1]
    if new_r > recon_r:
        selection[expl_var_ranks[i]] = True
        recon_vect = new_recon_vect
        recon_r = new_r
features['selected'] = selection

## display result
reconstructed = recon_vect.reshape([size, size])
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
axes[0].imshow(reconstructed, cmap='gray')
axes[1].imshow(image, cmap='gray', vmin=0, vmax=1)
plt.show()
