#!/usr/bin/env python
# -*- coding: utf-8 -*-
# peakfinder.py
"""
Class for finding blobs. Encapsulates a difference of gaussians (DoG)
algorithm and exposes methods to easilyt interact with the data and
results.

Copyright (c) 2016, David Hoffman
"""

import logging

# need math log too, for arbitrary base
from math import log

import dask

# we need a few extra features from matplot lib
import matplotlib.pyplot as plt

# Get our numerical stuff
import numpy as np

# need pandas for better data containers
import pandas as pd
import tqdm

# plotting
from dphplotting import display_grid
from dphutils import fft_gaussian_filter, mode, slice_maker

# ndimage imports
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
    median_filter,
    minimum_filter,
    uniform_filter1d,
)
from scipy.ndimage.measurements import find_objects, label

# specialty numpy and scipy imports
from scipy.signal import argrelmax
from scipy.spatial import cKDTree
from skimage._shared.utils import check_nD

# the difference of Gaussians algorithm
from skimage.draw import circle
from skimage.feature.peak import peak_local_max
from skimage.util import img_as_float

# import our 2D gaussian fitting class
from .gauss2d import Gauss2D, Gauss2Dz

logger = logging.getLogger(__name__)

from dask.diagnostics import ProgressBar


class PeakFinder(object):
    """
    A class to find peaks in image data and then fit them.

    Peak finder takes 2D data that is assumed to be made up of relatively
    sparse, approximately gaussian peaks. To estimate the positions of the
    peaks the [difference of Gaussians](https://en.wikipedia.org/wiki/Difference_of_Gaussians)
    algorithm is used as implemented in `skimage`. Once peaks have been found
    they are fit to a Gaussian function using the `Gauss2D` class in this
    package. Peak data is saved in a pandas DataFrame

    Parameters
    ----------
        data : ndarray
            2D data containing sparse gaussian peaks, ideally any background
            should be removed prior to construction
        sigma : float, optional, default: 1.0
            the estimated width of the peaks
    """

    def __init__(self, data, sigma=1.0, background="median"):
        # some error checking
        if not isinstance(data, np.ndarray):
            raise TypeError("data is not a numpy array")

        if data.ndim != 2:
            raise ValueError("The parameter `data` must be a 2-dimensional array")

        self._data = data
        # make an initial guess of the threshold
        if isinstance(background, str):
            self.estimate_background(background)
        else:
            self.thresh = background
        self._blobs = None
        # estimated width of the blobs
        self._blob_sigma = sigma

        self._labels = None
        # peak coefs from fits
        self._fits = None

    ########################
    # PROPERTY DEFINITIONS #
    ########################

    @property
    def data(self):
        """
        The data contained in the PeakFinder object
        """

        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._data

    @property
    def fits(self):
        """Optimized parameters from the fit"""
        # User should not be able to modify this, so return copy
        return self._fits.copy()

    @property
    def blobs(self):
        """Estimated peak locations"""
        # User should not be able to modify this, so return copy
        # sort blobs by the max amp value, descending
        blobs = self._blobs
        return blobs[blobs[:, -1].argsort()][::-1]

    @blobs.setter
    def blobs(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Blobs must be an ndarray")

        if value.ndim != 2:
            raise TypeError("Blobs don't have the right dimensions")

        if value.shape[-1] != 4:
            raise TypeError("Blobs don't have enough variables")

        # use a copy so that changes on the outside don't affect the internal
        # variable
        self._blobs = value.copy()

    @property
    def labels(self):
        """
        Estimated peak locations
        """
        # User should not be able to modify this, so return copy
        return self._labels.copy()

    @property
    def thresh(self):
        """Threshold for peak detection"""
        return self._thresh

    @thresh.setter
    def thresh(self, value):
        self._thresh = value

    @property
    def blob_sigma(self):
        """Estimated Peak width"""
        return self._blob_sigma

    @blob_sigma.setter
    def blob_sigma(self, value):
        self._blob_sigma = value

    ###########
    # Methods #
    ###########

    def estimate_background(self, method="median"):
        """Estimate the background/threshold of the data

        Two methods are available:
        - "median" : calculates median value of data as thresh
        - "mode" : if the data type is inexact it uses a histogram
            to estimate the mode, if the data is an unsigned integer
            then it uses `bincount`

        The result of the method is that the `thresh` property is set
        for the instance.
        """
        if method == "median":
            self.thresh = np.median(self.data)
        elif method == "mode":
            if np.issubdtype(self.data.dtype, np.inexact):
                hist, bins = np.histogram(self.data.ravel(), "auto")
                maxval = hist.argmax()
                # choose center of bin, not edges
                self.thresh = (bins[maxval] + bins[maxval + 1]) / 2
            elif np.issubdtype(self.data.dtype, np.unsignedinteger):
                self.thresh = mode(self.data)
            else:
                raise TypeError("Invalid type for method 'mode' {}".format(self.data.dtype))
        else:
            raise ValueError("Invalid option for `method`: {}".format(method))
        logger.debug("Threshold = {}".format(self.thresh))

    def find_blobs(self, method="dog", **kwargs):
        """
        Estimate peak locations by using a difference of Gaussians algorithm

        Parameters
        ----------
        min_sigma : float
            smallest sigma for DOG

        Returns
        -------
        blobs : ndarray
            blob parameters ordered as `y`, `x`, `sigma`, `amp`
        """
        # cast to float
        data = self.data.astype(float)
        # take care of the default kwargs with 'good' values
        default_kwargs = {
            "min_sigma": self.blob_sigma / np.sqrt(1.6),
            "max_sigma": self.blob_sigma * np.sqrt(1.6) * 0.9,
            "threshold": self.thresh,
        }

        # update default_kwargs with user passed kwargs
        default_kwargs.update(kwargs)

        # double check sigmas
        if default_kwargs["min_sigma"] >= default_kwargs["max_sigma"]:
            default_kwargs["max_sigma"] = default_kwargs["min_sigma"]

        # Perform the DOG
        if method.lower() == "dog":
            # NOTE: the threshold for `blob_dog` is the threshold in scale
            # space i.e. the threshold is not intuitively clear.
            blobs = better_blob_dog(data, **default_kwargs)
        else:
            raise NotImplementedError

        # if no peaks found alert the user, but don't break their program
        if blobs is None or len(blobs) == 0:
            logger.warning("No peaks found")

        else:
            # blobs, as returned, has the third index as the estimated width
            # for our application it will be beneficial to have the intensity
            # at the estimated center as well

            footprint = np.round(self.blob_sigma * 5)
            max_img = maximum_filter(data, footprint)
            # we just use mode, faster and more accurate for low
            # background images.
            diff_img = max_img - mode(data.astype(int))

            y, x, s = blobs.T

            blobs = np.vstack((y, x, s, diff_img[y.astype(int), x.astype(int)])).T

        self._blobs = blobs
        return self.blobs

    def label_blobs(self, diameter=None):
        """
        This function will create a labeled image from blobs
        essentially it will be circles at each location with diameter of
        4 sigma
        """

        tolabel = np.zeros_like(self.data)
        try:
            blobs = self.blobs
        except AttributeError:
            # try to find blobs
            blobs = self.find_blobs()
            # if blobs is still none, exit
            if blobs is None:
                logger.warning("Labels could not be generated")
                return None

        # Need to make this an ellipse using both sigmas and angle
        for blob in blobs:
            if diameter is None:
                radius = blob[2] * 4
            else:
                radius = diameter
            rr, cc = circle(blob[0], blob[1], radius, self._data.shape)
            tolabel[rr, cc] = 1

        labels, num_labels = label(tolabel)
        if num_labels != len(blobs):
            logger.warning("Blobs have melded, fitting may be difficult")

        self._labels = labels

        return labels

    def plot_blob_grid(self, window=11, **kwargs):
        """Display a grid of blobs"""
        return display_grid(
            {
                i: self.data[slice_maker((y, x), window)]
                for i, (y, x, s, r) in enumerate(self.blobs)
            },
            **kwargs
        )

    def plot_fits(self, window_width, residuals=False, **kwargs):
        """Generate a plot of the found peaks, individually"""

        # check if the fitting has been performed yet, warn user if it hasn't
        if self._fits is None:
            raise RuntimeError("Blobs have not been fit yet, cannot show fits")
        else:
            fits = self._fits

        # pull the labels and the data from the object
        data = self.data

        # find objects from labelled data
        my_objects = [slice_maker(center, window_width) for center in fits[["y0", "x0"]].values]

        # generate a nice layout
        nb_labels = len(my_objects)

        nrows = int(np.ceil(np.sqrt(nb_labels)))
        ncols = int(np.ceil(nb_labels / nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

        for n, (obj, ax) in enumerate(zip(my_objects, axes.ravel())):
            ex = (obj[1].start, obj[1].stop - 1, obj[0].stop - 1, obj[0].start)
            ax.set_title(n)
            ax.grid("off")

            # generate the model fit to display, from parameters.
            dict_params = dict(fits.loc[n].dropna())

            # recenter
            dict_params["x0"] -= obj[1].start
            dict_params["y0"] -= obj[0].start
            params = Gauss2D.dict_to_params(dict_params)
            fake_data = Gauss2D.gen_model(data[obj], *params)
            if residuals:
                ax.matshow(data[obj] - fake_data, extent=ex, **kwargs)
            else:
                ax.matshow(data[obj], extent=ex, **kwargs)
                ax.contour(fake_data, extent=ex, colors="w", origin="image")

        # # Remove empty plots
        for ax in axes.ravel():
            if not (len(ax.images)) and not (len(ax.lines)):
                fig.delaxes(ax)

        fig.tight_layout()

        # return the fig and axes handles to user for later manipulation.
        return fig, axes

    def filter_blobs(self, minamp=None, maxamp=None):
        amps = self.blobs[:, 3]
        if maxamp is None:
            maxamp = amps.max()

        if maxamp is None:
            minamp = amps.min()

        self.blobs = self.blobs[np.logical_and(maxamp > amps, amps > minamp)]

        return self.blobs

    def fit_blobs(self, width=10, poly_coefs_df=None, **kwargs):
        """Fit blobs to Gaussian funtion.
        Parameters
        ----------
        width : int
            The size of the fitting window in pixels

        **kwargs is for Gauss2D optimize_params
        """
        # If we don't have blobs, find them.
        if self._blobs is None:
            self.find_blobs()

        @dask.delayed
        def fitfunc(win, sub_data):
            # fit the data as we should
            if poly_coefs_df is None:
                mypeak = Gauss2D(sub_data)
            else:
                mypeak = Gauss2Dz(sub_data, poly_coefs_df)
            # optimize params
            mypeak.optimize_params(**kwargs)
            fit_coefs = mypeak.all_params_dict()
            # need to place the fit coefs in the right place
            fit_coefs["y0"] += win[0].start
            fit_coefs["x0"] += win[1].start
            # Calc SNR for each peak
            fit_coefs["noise"] = mypeak.noise
            fit_coefs["SNR"] = fit_coefs["amp"] / fit_coefs["noise"]
            return fit_coefs

        # iterate through blobs
        windows = [slice_maker((int(y), int(x)), width) for y, x, s, r in self.blobs]
        data_to_fit = [self.data[win] for win in windows]
        peakfits = dask.delayed(
            [fitfunc(win, sub_data) for win, sub_data in zip(windows, data_to_fit)]
        )
        # construct DataFrame
        peakfits_df = pd.DataFrame(peakfits.compute())
        # internalize DataFrame
        self._fits = peakfits_df
        # Return it to user
        return peakfits_df

    fit_blobs.__doc__ += Gauss2D.optimize_params.__doc__

    def prune_blobs(self, radius):
        """
            Pruner method takes blobs list with the third column replaced by
            intensity instead of sigma and then removes the less intense blob
            if its within diameter of a more intense blob.

            Adapted from _prune_blobs in skimage.feature.blob

            Parameters
            ----------
            blobs : ndarray
                A 2d array with each row representing 3 values,
                `(y, x, intensity)` where `(y, x)` are coordinates
                of the blob and `intensity` is the intensity of the
                blob (value at (x, y)).
            diameter : float
                Allowed spacing between blobs

            Returns
            -------
            A : ndarray
                `array` with overlapping blobs removed.
            """

        # make a copy of blobs otherwise it will be changed
        # create the tree
        blobs = self.blobs
        kdtree = cKDTree(blobs[:, :2])
        # query all pairs of points within diameter of each other
        list_of_conflicts = list(kdtree.query_pairs(radius))
        # sort the collisions by max amplitude of the pair
        # we want to deal with collisions between the largest
        # blobs and nearest neighbors first:
        # Consider the following sceneario in 1D
        # A-B-C
        # are all the same distance and colliding with amplitudes
        # A > B > C
        # if we start with the smallest, both B and C will be discarded
        # If we start with the largest, only B will be
        # Sort in descending order
        list_of_conflicts.sort(key=lambda x: max(blobs[x[0], -1], blobs[x[1], -1]), reverse=True)
        # indices of pruned blobs
        pruned_blobs = set()
        # loop through conflicts
        for idx_a, idx_b in list_of_conflicts:
            # see if we've already pruned one of the pair
            if (idx_a not in pruned_blobs) and (idx_b not in pruned_blobs):
                # compare based on amplitude
                if blobs[idx_a, -1] > blobs[idx_b, -1]:
                    pruned_blobs.add(idx_b)
                else:
                    pruned_blobs.add(idx_a)
        # generate the pruned list
        # pruned_blobs_set = {(blobs[i, 0], blobs[i, 1])
        #                         for i in pruned_blobs}
        # set internal blobs array to blobs_array[blobs_array[:, 2] > 0]
        self._blobs = blobs[[i for i in range(len(blobs)) if i not in pruned_blobs]]
        # Return a copy of blobs incase user wants a one-liner
        return self.blobs

    def remove_edge_blobs(self, distance):
        """Remove blobs that are less than `distance` away from the image
        edge"""
        # find the maximum limits of the data
        ymax, xmax = self._data.shape
        # build a new array filtering out any blobs that are two close to
        # the edge of the image
        my_blobs = np.array(
            [
                blob
                for blob in self.blobs
                if (
                    (distance < blob[0] < ymax - distance)
                    and (distance < blob[1] < xmax - distance)
                )
            ]
        )
        # resort the blobs, largest to smallest
        if len(my_blobs) > 0:
            my_blobs = my_blobs[my_blobs[:, 3].argsort()]
        # set the internals and return them
        self._blobs = my_blobs
        return self.blobs

    def plot_blobs(self, diameter=None, size=6, with_labels=True, **kwargs):
        """Plot the found blobs

        Parameters
        ----------
        diameter : numeric
            diameter of the circles to draw, if omitted
            the diameter will be 4 times the estimated
            sigma
        size : int
            The size of the final plot
        **kwargs : key word arguments
            Any extra keyword arguments are passed along to plt.matshow

        Returns
        -------
        fig, axs : plt.figure, ndarray of plt.axes
        """
        if self.blobs is None:
            raise RuntimeError("No blobs have been found")

        ny, nx = self.data.shape
        fig, ax = plt.subplots(1, 1, figsize=(size, size * ny / nx))

        ax.matshow(self.data, **kwargs)

        if with_labels:
            for i, blob in enumerate(self.blobs):
                y, x, s, r = blob
                if diameter is None:
                    diameter = s * 4

                c = plt.Circle(
                    (x, y),
                    radius=diameter / 2,
                    color="r",
                    linewidth=1,
                    fill=False,
                    transform=ax.transData,
                )
                ax.add_patch(c)

                if not np.issubdtype(float, self.data.dtype):
                    r = int(r)
                    fmtstr = "{}"
                else:
                    fmtstr = "{}:{:.0f}"

                ax.annotate(
                    fmtstr.format(i, r),
                    xy=(x, y),
                    xytext=(x + diameter / 2, y + diameter / 2),
                    textcoords="data",
                    color="k",
                    backgroundcolor=(1, 1, 1, 0.5),
                    xycoords="data",
                )
        else:
            ax.scatter(
                self.blobs[:, 1],
                self.blobs[:, 0],
                s=self.blobs[:, 2] * 10,
                marker="o",
                facecolor="none",
                edgecolor="w",
            )

        return fig, ax


def better_blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=0.03):
    """Finds blobs in the given grayscale image.
    Blobs are found using the Difference of Gaussian (DoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    sigma_ratio : float, optional
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y, x, sigma)``
        where ``(y, x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Blob_detection# The_difference_of_Gaussians_approach
    Notes
    -----
        The radius of each blob is approximately :math:`\sqrt{2}sigma`.
    """
    check_nD(image, 2)

    image = img_as_float(image)
    sigma_ratio = float(sigma_ratio)
    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i) for i in range(k + 1)])

    # Use the faster fft_gaussian_filter to speed things up.
    gaussian_images = [fft_gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1]) * sigma_list[i] for i in range(k)]
    image_cube = np.dstack(dog_images)
    # peak_local_max is looking in the image_cube, so threshold should
    # be scaled by differences in sigma, i.e. sigma_ratio
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        footprint=np.ones((3, 3, 3)),
        threshold_rel=0.0,
        exclude_border=False,
    )
    if local_maxima.size:
        # Convert local_maxima to float64
        lm = local_maxima.astype(np.float64)
        # Convert the last index to its corresponding scale value
        lm[:, 2] = sigma_list[local_maxima[:, 2]]
        local_maxima = lm
    return local_maxima


##############################################################################
#                          Spectral Peak Finding Part                        #
##############################################################################


class SpectralPeakFinder(object):
    """
    A class used to find peaks in data that has one spatial and one spectral
    and one time dimension

    Data is assumed to have dimensions time (0), space (1), spectral (2)
    """

    # NOTE that the way this class is implemented it does not hide any of its
    # variables or methods from the user.

    def __init__(self, data):
        """
        A class designed to find peaks in spectral/spatial/time data
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data is not a numpy array")

        # this is **VERY** data _un_aware!
        # this makes a copy, which means that original data should be safe
        # we're casting to a signed 32 bit int which has enough bit depth to
        # accomodate the original data (uint16) but also allows negative
        # numbers.
        self.data = data.astype(int)
        self.peaks = None

    def remove_background(self):
        """
        Remove background from the data cube.

        This method uses a relatively simple algorithm that first takes the
        mean along the time dimension and then the median along the spatial
        dimension

        The assumption here is that peaks are relatively sparse along the
        spatial dimension

        NOTE: This function mutates the data internally
        """
        # pull internal data
        data = self.data
        # take the median value along the time and spatial dimensions
        # keep the dimensions so that broadcasting will work properly

        # bg = np.median(data, axis=(0, 1), keepdims=True)
        # this is much faster than the above but gives approximately the same
        # results
        bg = np.median(data.mean(0), 0)

        self.data = data - bg

    def fix_hot_pixels(self, cutoff=9):
        """
        A method to remove "Salt and Pepper" noise from the image stack

        This method assumes that hot pixels do not vary much with time and uses
        this property to avoid performing a median filter for every time point.

        Remember this function mutates the data internally
        """
        # pull internal data
        data = self.data

        # calc the _mean_ projection
        # the assumption is that if the hot pixel is in one frame it will be
        # in all of them and the whole point of this method is to only perform
        # the median filter once
        mean_data = data.mean(0)

        # do the one median filter, use a 3x3 footprint
        # some articles suggest that a 2x2 is fine, but I'm not sure if I buy
        # that
        # NOTE: that because we're filtering _single_ pixels
        mean_data_med = median_filter(mean_data, 3)

        # subtract the median filtered data from the unfiltered data
        data_minus = mean_data - mean_data_med

        # calculate the z-score for each pixel
        z_score = (data_minus - data_minus.mean()) / data_minus.std()

        # find the points to remove
        picked_points = (z_score > cutoff) * mean_data

        # remove them from the data
        data -= picked_points

        # return the number of points removed
        return np.count_nonzero(picked_points)

    def fix_cosmic_rays(self, width, z_score_cutoff=2.5):
        """
        Method to remove cosmic rays from good peaks.

        Assumes that cosmic rays only show up for one frame and are *bright*
        """
        # calculate the average around the peaks
        mean_data_sum = uniform_filter1d(self.data, width, axis=1).sum(2)
        z_score = (mean_data_sum.max(0) - mean_data_sum.mean(0)) / mean_data_sum.std(0)
        bad_peaks = np.arange(len(z_score))[z_score > z_score_cutoff]

        self.peaks = [p for p in self.peaks if p not in bad_peaks]

    def calc_FoM(self, width, s_lambda=3, s_time=3, use_max=False):
        """
        Calculate the figure of merit (FoM) of a dataset (t, x, and lambda)

        In this case our figure of merit is calculated as the _maximum_ value
        along the spectral dimension aver the

        Parameters
        ----------
        data : ndarray (NxMxK)
            the array overwhich to calculate the SNR, assumes that it
            has dimensions (time, position, spectrum)
        width : int
            the width overwhich to calculate the average in the spatial
            dimension
        s_lambda : float (optional)
            the width of the gaussian kernel along the spectral dimension
        s_time : float (optional)
            the width of the gaussian kernel along the time dimension
        use_max : bool (optional)
            whether to use the max projection or not, will significantly speed
            up the calculation but will raise the noise floor in the process.

        Returns
        -------
        FoM : ndarray (NxK)
            The calculated figure of merit (FoM)
        """

        # before we make another copy we should trash the old one, if it exists
        # if we don't do this it can lead to a memory leak.
        try:
            del self.g_mean_data
        except AttributeError:
            pass

        # First calculate the moving average of the data along the spatial
        # dimension cast as float64 for better precision, this is necessary
        # for the later gaussian filters, but might as well do it now to avoid
        # making more copies of the data than necessary.

        if use_max:
            data = self.data.max(0, keepdims=True).astype(float)
        else:
            data = self.data.astype(float)

        mean_data = uniform_filter1d(data, width, axis=1)

        # calculate the gaussian blue along the spectral and time dimensions
        if s_time == 0 and s_lambda == 0:
            g_mean_data = mean_data
        else:
            g_mean_data = gaussian_filter(mean_data, (s_time, 0, s_lambda))

        g_mean_data_mean = g_mean_data.mean(axis=(0, 2))
        g_mean_data_std = g_mean_data.std(axis=(0, 2))
        g_mean_data_max = g_mean_data.max(axis=(0, 2))

        FoM = (g_mean_data_max - g_mean_data_mean) / g_mean_data_std

        self.FoM = FoM
        self.g_mean_data = g_mean_data

    def find_peaks(self, width, cutoff=7, cutoff_high=np.inf, presmooth=0, show=False):
        """
        A function that finds peaks in the FoM trace.
        """

        # find the local maxima in the SNR trace
        # presmooth might make sense here
        if presmooth:
            FoM = gaussian_filter(self.FoM, presmooth)
            width2 = int(2 * presmooth * np.sqrt(2 * np.log(2)))
        elif presmooth is None:
            FoM = gaussian_filter(self.FoM, width * (np.sqrt(2 * np.log(2))))
            width2 = int(2 * width * (2 * np.log(2)))
        else:
            FoM = self.FoM
            width2 = width
        peaks = argrelmax(FoM * (FoM > cutoff), order=width)[0]
        # here we look to see the *relative* intensity of the peak.
        # set up our container
        good_peaks = []
        for p in peaks:
            # find the lower side
            pm = max(p - width2, 0)
            # find the upper side
            pp = min(p + width2, len(FoM) - 1)
            # test if peak minus sides is within cutoff
            # Below tests a *relative* cutoff
            # should test an absolute cutoff as well
            if FoM[p] - min(FoM[pm], FoM[pp]) > cutoff:
                # if not, add peak
                good_peaks.append(p)
        # peaks = peaks[FoM[peaks] < cutoff_high]

        # Show the peaks?
        if show:
            fig, ax = plt.subplots(1, 1)
            ax.plot(FoM)
            ax.plot(good_peaks, FoM[good_peaks], "ro")
            ax.axis("tight")

        self.peaks = good_peaks

    def refine_peaks(self, window_width=8):
        """
        A function that refines peaks.

        Because of the way the FoM is calculated the highest SNR region isn't
        identified because the noise is approximated by the std. This function
        will search the nearby are for a peak (using the smoothed data) and
        will return that point instead.

        Parameters
        ----------
        window_width : int (optional)
            the window in which to search for a peak.
        """
        new_peaks = []

        # take the max of the data along the time axis
        max_data = self.g_mean_data.max(0)
        ny, nx = max_data.shape

        ny = window_width * 2

        # NOTE: this implementation is pretty slow. But I'm not quite sure how
        # to speed it up.
        for peak in self.peaks:
            # find the max
            dy, dx = np.unravel_index(
                max_data[peak - window_width : peak + window_width].argmax(), (ny, nx)
            )
            new_peaks.append(peak - window_width + dy)

        self.peaks = np.array(new_peaks)

    def _plot_peaks_lines(self):
        """
        A helper function to plot a max intensity projection with redlines
        marking the location of the found peaks.
        """
        figmat, axmat = plt.subplots(1, 1, squeeze=True, sharex=True)
        axmat.matshow(self.data.max(0))
        axmat.set_yticks(self.peaks)
        for peak in self.peaks:
            axmat.axhline(peak, color="r")

    def plot_peaks(self):
        """
        A utility function to plot the found peaks.
        """

        peaks = self.peaks
        FoM = self.FoM
        g_mean_data = self.g_mean_data
        nz, ny, nx = g_mean_data.shape
        # plot the found peaks in the SNR trace

        print(g_mean_data.shape)

        # self._plot_peaks_lines()

        for peak in peaks:

            # need to ensure a reasonable ratio
            ratio = nz / nx
            if ratio < 0.05:
                ratio = 0.05

            fig, (ax0, ax1) = plt.subplots(
                2, 1, squeeze=True, sharex=True, figsize=(12, 12 * ratio * 2)
            )

            ax0.matshow(g_mean_data[:, peak, :])
            ax0.axis("tight")
            ax0.set_xticks([])

            ax1.plot(g_mean_data[:, peak, :].max(0))
            ax1.axis("tight")

            fig.suptitle("{}, Max SNR {:.3f}".format(peak, FoM[peak]), y=1, fontsize=14)

            fig.tight_layout()


class SpectralPeakFinder1d(SpectralPeakFinder):
    """
    A class to find peaks in a single frame.
    """

    def __init__(self, data):
        # reshape the data so that it can use the previous methods without
        # changes
        super().__init__(data.reshape(1, *data.shape))

    # overload the plot peaks function
    def plot_peaks(self):
        """
        A utility function to plot the found peaks.
        """

        peaks = self.peaks
        FoM = self.FoM
        g_mean_data = self.g_mean_data
        nz, ny, nx = g_mean_data.shape
        # plot the found peaks in the SNR trace

        self._plot_peaks_lines()

        data_dict = {
            "{}, Max SNR {:.3f}".format(peak, FoM[peak]): g_mean_data[0, peak, :] for peak in peaks
        }
        return display_grid(data_dict)

    def fix_cosmic_rays(self, *args, **kwargs):
        """
        This method is invalid for this type of data
        """

        raise ValueError("This method is not valid for 1d data.")
