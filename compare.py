import cv2
import numpy
from numpy import pi
import scipy.signal
import matplotlib.pyplot as plt
plt.ion()


img = cv2.imread('images/dog_200.png')   ## 3 channels, uint8
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## 1 channel, uint8

## show image
plt.figure()
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
# %timeit


"""The key is to adapt the sigma and the kernel size to the lambda. Together they make up the SF
"""

ksize = (121, 121)    ##  Size of the filter returned.
sigma = 20           ##  Standard deviation of the gaussian envelope.
theta = 1 * pi/4    ##  Orientation of the normal to the parallel stripes of a Gabor function.
lambd = 60 #60 # * pi/4   ##  Wavelength of the sinusoidal factor.
gamma = 0.5         ##  Spatial aspect ratio. ("ellipsicity") 1 is round, 0 straight line
psi = 0             ##  Phase offset.
ktype = cv2.CV_32F  ##  Type of filter coefficients. It can be CV_32F or CV_64F .(float32)


kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)

## display kernel
plt.figure()
plt.imshow(kernel, cmap='gray')
plt.show()

## filter
filt_img = cv2.filter2D(img, -1, kernel)

## display filtered img
plt.figure()
plt.imshow(filt_img, cmap='gray', vmin=0, vmax=255)
plt.show()

# filterdepth code explainer: https://stackoverflow.com/a/27184054/708221

# scipy.signal.convolve2d(similarity_matrix, np.diag(filter), mode="same")


# scipy.signal.fftconvolve(face, kernel, mode='same')


plt.ioff()
