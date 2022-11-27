import numpy as np
import cv2
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def blur(i, k):
    i1 = np.copy(i)
    factor = 1 / k
    modifiedDiagnolMatrix = factor * np.eye(k)  # np.eye(k) forms a diagnol k*k matrix
    return convolve2d(i1, modifiedDiagnolMatrix, mode='valid')


def addGaussNoise(i, standev):
    mean = 0
    gauss = np.random.normal(mean, standev, np.shape(i))  # random gaussian probablity distribution
    noisyimage = i + gauss
    return noisyimage


def gaussian(M, std, sym=True):
    # Return a Gaussian window.
    # Parameters
    # ----------
    # M : int
    #     Number of points in the output window. If zero or less, an empty
    #     array is returned.
    # std : float
    #     The standard deviation, sigma.
    # sym : bool, optional
    #     When True (default), generates a symmetric window, for use in filter
    #     design.
    #     When False, generates a periodic window, for use in spectral analysis.
    # Returns
    # w : ndarray
    #     The window, with the maximum value normalized to 1 (though the value 1
    #     does not appear if `M` is even and `sym` is True).

    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)  #
    h = np.dot(h, h.transpose())  # dot prod of gaussian matrix and its transpose
    h1 = h/np.sum(h)  # np.sum(h) computes the sum of all the elements in the matrix h
    return h1


def wiener_filter(img, kernel, K):
    kernel = kernel/np.sum(kernel)  # the kernel gets divided by the sum of its elements
    imgcopy = np.copy(img)
    fourierimg = fft2(imgcopy)  # we take fft of image
    kernel = fft2(kernel, s=img.shape)  # Shape (length of each transformed axis) of the output. Along each axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros. if s is not given, the shape of the input along the axes specified by axes is used.
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)  # formula computation
    convolvedimg = fourierimg * kernel  # we convolve the two signals in the frequency domain
    invfourier = np.abs(ifft2(convolvedimg))  # compute inv fourier
    return invfourier


if __name__ == '__main__':

    path = '/Users/kabirkumar/Desktop/Sem 3/ELL205/img.jpeg'

    # Original Image
    img = cv2.imread(path)

    # Grayscale Image
    grayscale_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Blur the image
    blurred_img = blur(grayscale_img, 70)

    # Add Gaussian noise
    noisy_img = addGaussNoise(blurred_img, 20)

    # Apply Wiener Filter
    kernel = gaussian_kernel(5)
    filtered_img = wiener_filter(noisy_img, kernel, 10)

    # Display results
    display = [grayscale_img, blurred_img, noisy_img, filtered_img]
    label = ['Original Image', 'Motion Blurred Image', 'Motion Blurring + Gaussian Noise', 'Wiener Filter applied']

    fig = plt.figure(figsize=(12, 10))

    for i in range(len(display)):
        fig.add_subplot(2, 2, i + 1)
        plt.imshow(display[i], cmap='gray')
        plt.title(label[i])

    plt.show()
