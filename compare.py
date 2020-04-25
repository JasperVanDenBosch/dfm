"""The key is to adapt the sigma and the kernel size to the lambda. Together they make up the SF

very useful: http://matlabserver.cs.rug.nl/edgedetectionweb/web/edgedetection_params.html

scale of kernel values wrt image also important

todo:
- restrict theta to 180 degrees

TO READ: https://stackoverflow.com/q/61317974/708221


"""

import cv2
import numpy
from numpy import pi, ceil
import scipy.signal
import matplotlib.pyplot as plt
from os.path import join
import itertools

plotdir = 'plots'
img = cv2.imread('images/dog_200.png')   ## 3 channels, uint8
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255 ## 1 channel

n_sfs = 10
n_oris = 8
start = 1.5
stop = 40
# sig_factor = 0.3 # 0.56
thetas = numpy.arange(n_oris) * (pi/4)
sfs = numpy.geomspace(start, stop, num=10)
size = img.shape[0]

features = list(itertools.product(thetas, sfs))
for f, (theta, sf) in enumerate(features):
    print(f'{f+1}/{len(features)}')

    ## parameters
    theta = theta           ##  Orientation of the normal to the parallel stripes of a Gabor function.
    lambd = size/sf         ##  Wavelength of the sinusoidal factor.
    sigma = 0.56*lambd  # 0.56   ##  Standard deviation of the gaussian envelope. 0.56*lambd
    kside = 1 + (16 * int(ceil(lambd))) ##  Size of the filter returned. should be a fn of sigma & lambda
    gamma = 0.5             ##  Spatial aspect ratio. ("ellipsicity") 1 is round, 0 straight line
    psi = 0                 ##  Phase offset.
    ktype = cv2.CV_32F      ##  Type of filter coefficients. It can be CV_32F or CV_64F .(float32)

    kernel = cv2.getGaborKernel((kside, kside), sigma, theta, lambd, gamma, psi, ktype)
    kernel /= numpy.sqrt((kernel * kernel).sum())

    ## filter
    filt_img = cv2.filter2D(img, -1, kernel)
    #filt_img = scipy.signal.fftconvolve(img, kernel, mode='same')
    # raise ValueError

    ## display filtered img
    theta_degrees = int(numpy.degrees(theta))
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
    kernel_imsize = numpy.zeros([size, size])
    offset = abs(int((size-(kside-1))/2)) ## this only works if size is even
    if kside < size:
        kernel_imsize[offset-2:-offset+1, offset-2:-offset+1] = kernel.max()
        kernel_imsize[offset-1:-offset, offset-1:-offset] = kernel
    else:
        kernel_imsize = kernel[offset:-offset, offset:-offset]
    axes[0].imshow(kernel_imsize, cmap='gray', vmin=0)
    axes[1].imshow(filt_img, cmap='gray', vmin=0, vmax=1) # , vmin=0, vmax=255
    axes[2].imshow(img*255, cmap='gray', vmin=0, vmax=255)
    fig.suptitle(f'{theta_degrees}°  {sf:.1f} cycles')
    plt.savefig(join(plotdir, f'{f+1}.png'))
    plt.close(fig)

# filterdepth code explainer: https://stackoverflow.com/a/27184054/708221
# scipy.signal.convolve2d(similarity_matrix, np.diag(filter), mode="same")
# scipy.signal.fftconvolve(face, kernel, mode='same')

