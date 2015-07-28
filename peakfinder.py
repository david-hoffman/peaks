#Copyright David P. Hoffman

'''
A Class to find peaks and fit them
'''

import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max

#we need a few extra features from matplot lib
from matplotlib.path import Path #Needed to create shapes
import matplotlib.patches as patches #needed so show shapes on top of graphs
import matplotlib.gridspec as gridspec #fancy subplot layout

import warnings

import .Gauss2D

class PeakFinder(object):
    """
    A class to find peaks in image data and then fit them.
    """
    def __init__(self, data, sigma = 1.0, modeltype = 'sym'):
        self._data = data
        #make an initial guess of the threshold
        self._thresh = np.median(data)
        self._blobs = None
        #these are the fit coefficients
        self._peak_coefs = None
        self._modeltype = modeltype
        #estimated width of the blobs
        self._blob_sigma = sigma

    ########################
    # PROPERTY DEFINITIONS #
    ########################

    @property
    def data(self):
        '''
        Optimized parameters from the fit
        '''

        #This attribute should be read-only, which means that it should return
        #a copy of the data not a pointer.
        return self._data

    @property
    def peak_coefs(self):
        '''
        Optimized parameters from the fit
        '''
        #User should not be able to modify this, so return copy
        return self._peak_coefs.copy()

    @property
    def blobs(self):
        '''
        Optimized parameters from the fit
        '''
        #User should not be able to modify this, so return copy
        return self._blobs.copy()

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

    def estimate_background(self):
        raise NotImplementedError('estimate_background')

    def find_blobs(self):
        raise NotImplementedError('find_blobs')

    def fit_blobs(self):

    def prune_blobs(self):

    def plot_blobs(self, diameter = self.blob_sigma*10):

        if self.blobs is None
            raise UserWarning('No blobs have been found')

        fig, ax = plt.subplots(1, 1,figsize=(12,12))

        ax.matshow(self.data)
        for blob in self.blobs:
            y, x, r = blob
            c = plt.Circle((x, y), radius=diameter/2, color='r', linewidth=2,
                           fill=False)
            ax.add_patch(c)
            ax.annotate('{:.3f}'.format(r), xy=(x, y),xytext = (5,5),\
                textcoords = 'offset points', color='k', backgroundcolor=(1,1,1,0.5) ,xycoords='data')

        return fig, ax
