"""The key is to adapt the sigma and the kernel size to the lambda. Together they make up the SF

very useful: http://matlabserver.cs.rug.nl/edgedetectionweb/web/edgedetection_params.html
"""

import cv2
import numpy
from numpy import pi, ceil
import scipy.signal
import matplotlib.pyplot as plt
from os.path import join

plotdir = 'plots'
img = cv2.imread('images/dog_200.png')   ## 3 channels, uint8
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## 1 channel, uint8

# %timeit

n_sfs = 10
n_oris = 8
start = 1.2
stop = 40
thetas = numpy.arange(n_oris) * (pi/4)
sfs = numpy.geomspace(start, stop, num=10)
size = img.shape[0]

features = list(zip(thetas, sfs))
for f, (theta, sf) in enumerate(features):


    ## parameters
    theta = theta           ##  Orientation of the normal to the parallel stripes of a Gabor function.
    lambd = size/sf         ##  Wavelength of the sinusoidal factor.
    sigma = 0.56 * lambd    ##  Standard deviation of the gaussian envelope.
    kside = 1 + (5 * int(ceil(lambd))) ##  Size of the filter returned.
    gamma = 0.5             ##  Spatial aspect ratio. ("ellipsicity") 1 is round, 0 straight line
    psi = 0                 ##  Phase offset.
    ktype = cv2.CV_32F      ##  Type of filter coefficients. It can be CV_32F or CV_64F .(float32)

    kernel = cv2.getGaborKernel((kside, kside), sigma, theta, lambd, gamma, psi, ktype)

    ## filter
    filt_img = cv2.filter2D(img, -1, kernel)

    ## display filtered img
    theta_degrees = int(numpy.degrees(theta))
    fig, axes = plt.subplots(nrows=1, ncols=3)
    kernel_imsize = numpy.zeros([size, size])
    offset = abs(int((size-(kside-1))/2)) ## this only works if size is even
    if kside < size:
        kernel_imsize[offset:-offset, offset:-offset] = kernel
    else:
        kernel_imsize = kernel[offset:-offset, offset:-offset]
    axes[0].imshow(kernel, cmap='gray')
    axes[1].imshow(filt_img, cmap='gray', vmin=0, vmax=255)
    axes[2].imshow(img, cmap='gray', vmin=0, vmax=255)
    fig.suptitle(f'{theta_degrees}°  {sf:.1f} cycles')
    plt.savefig(join(plotdir, f'{f}.png'))

# filterdepth code explainer: https://stackoverflow.com/a/27184054/708221
# scipy.signal.convolve2d(similarity_matrix, np.diag(filter), mode="same")
# scipy.signal.fftconvolve(face, kernel, mode='same')

