#Copyright David P. Hoffman

'''
A Class to find peaks and fit them
'''

#Get our numerical stuff
import numpy as np

#the difference of Gaussians algorithm
from skimage.feature import blob_dog
from skimage.draw import circle
from skimage.filters import threshold_yen
from skimage.util import img_as_float
from skimage.feature.peak import peak_local_max
from skimage.transform import integral_image
from skimage._shared.utils import assert_nD

#we need ittertools for the pruner function defined below
import itertools as itt

#we need a few extra features from matplot lib
import matplotlib.pyplot as plt
from matplotlib.path import Path #Needed to create shapes
import matplotlib.patches as patches #needed so show shapes on top of graphs

#We want to be able to warn the user about potential problems
import warnings

from scipy.ndimage import gaussian_filter, median_filter, uniform_filter1d
from scipy.ndimage.measurements import label, find_objects

#import our 2D gaussian fitting class
from .gauss2d import Gauss2D

#need pandas for better data containers
import pandas as pd

from numpy.linalg import norm
from scipy.signal import argrelmax

import math

class PeakFinder(object):
    '''
    A class to find peaks in image data and then fit them.

    Peak finder takes 2D data that is assumed to be made up of relatively sparse,
    approximately gaussian peaks. To estimate the positions of the peaks the
    [difference of Gaussians](https://en.wikipedia.org/wiki/Difference_of_Gaussians)
    algorithm is used as implemented in `skimage`. Once peaks have been found they
    are fit to a Gaussian function using the `Gauss2D` class in this package.
    Peak data is saved in a pandas DataFrame

    Parameters
    ----------
        data : ndarray
            2D data containing sparse gaussian peaks, ideally any background
            should be removed prior to construction
        sigma : float, optional, default: 1.0
            the estimated width of the peaks
        modeltype : ['sym' | 'norot' | 'full'], optional, default: 'sym'
            The peak model, see the documentation for `Gauss2D` for more info
    '''

    def __init__(self, data, sigma = 1.0, modeltype = 'sym'):
        #some error checking
        if not isinstance(data, np.ndarray):
            raise TypeError('data is not a numpy array')

        if data.ndim != 2:
            raise ValueError('The parameter `data` must be a 2-dimensional array')

        self._data = data
        #make an initial guess of the threshold
        self._thresh = np.median(data)
        self._blobs = None
        self._modeltype = modeltype
        #estimated width of the blobs
        self._blob_sigma = sigma

        self._labels = None
        #peak coefs from fits
        self._fits = None

    ########################
    # PROPERTY DEFINITIONS #
    ########################

    @property
    def data(self):
        '''
        The data contained in the PeakFinder object
        '''

        #This attribute should be read-only, which means that it should return
        #a copy of the data not a pointer.
        return self._data

    def modeltype():
        doc = "The modeltype property."
        def fget(self):
            return self._modeltype
        def fset(self, value):
            self._modeltype = value
        def fdel(self):
            del self._modeltype
        return locals()
    modeltype = property(**modeltype())

    @property
    def peak_coefs(self):
        '''
        Optimized parameters from the fit
        '''
        #User should not be able to modify this, so return copy
        return self._fits.copy()

    @property
    def blobs(self):
        '''
        Estimated peak locations
        '''
        #User should not be able to modify this, so return copy
        return self._blobs.copy()

    @blobs.setter
    def blobs(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError('Blobs must be an ndarray')

        if value.ndim != 2:
            raise TypeError("Blobs don't have the right dimensions")

        if value.shape[-1] != 4:
            raise TypeError("Blobs don't have enough variables")

        #use a copy so that changes on the outside don't affect the internal
        #variable
        self._blobs = value.copy()

    @property
    def labels(self):
        '''
        Estimated peak locations
        '''
        #User should not be able to modify this, so return copy
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


    def estimate_background(self):
        self.thresh = np.median(self.data)


    def find_blobs(self, type='dog', **kwargs):
        '''
        Estimate peak locations by using a difference of Gaussians algorithm

        Parameters
        ----------
        min_sigma : floatsmallest sigma for DOG

        '''

        #scale the data to the interval of [0,1], the DOG algorithm works best
        #for this

        data = self.data.astype(float)

        dmin = data.min()
        dmax = data.max()

        scaled_data = (data-dmin)/(dmax-dmin)

        #Threshold is a special variable because it needs to be scaled as well.
        #See if the user has tried to pass it
        try:
            thresh = kwargs.pop('threshold')
        except KeyError as e:
            #if that fails then set the threshold to the object's
            thresh = self.thresh

        #now scale the threshold the same as we did for the data
        scaled_thresh = (float(thresh)-dmin)/(dmax-dmin)
        #take care of the default kwargs with 'good' values
        default_kwargs = {'min_sigma' : self.blob_sigma/1.6, \
        'max_sigma' : self.blob_sigma*1.6, 'threshold' : scaled_thresh}#, 'overlap' : 0.0}

        #set default values for kwargs before passing along
        for k, v in default_kwargs.items():
            if k not in kwargs.keys():
                kwargs[k] = v

        #double check sigmas
        if kwargs['min_sigma'] >= kwargs['max_sigma']:
            kwargs['max_sigma'] = kwargs['min_sigma']*1.6**2

        #Perform the DOG
        if type.lower() == 'dog':
            #NOTE: the threshold for `blob_dog` is the threshold in scale space
            #i.e. the threshold is not intuitively clear.
            blobs = better_blob_dog(scaled_data,**kwargs)
        else:
            blobs = None

        #if no peaks found alert the user, but don't break their program
        if blobs is None or len(blobs) == 0:
            warnings.warn('No peaks found',UserWarning)

        else:
            #blobs, as returned, has the third index as the estimated width
            #for our application it will be beneficial to have the intensity at
            #the estimated center as well
            blobs = np.array([[y, x, s,self.data[y,x]] for y, x, s in blobs])

            #sort blobs by the max am value
            blobs = blobs[blobs[:,3].argsort()]

        self._blobs = blobs
        return blobs

    def label_blobs(self, diameter = None):
        '''
        This function will create a labeled image from blobs
        essentially it will be circles at each location with diameter of
        4 sigma
        '''

        tolabel = np.zeros_like(self.data)
        try:
            blobs = self.blobs
        except AttributeError as e:
            #try to find blobs
            blobs = self.find_blobs()
            #if blobs is still none, exit
            if blobs is None:
                warnings.warn('Labels could not be generated', UserWarning)
                return None

        #Need to make this an ellipse using both sigmas and angle
        for blob in blobs:
            if diameter is None:
                radius = blob[2]*4
            else:
                radius = diameter
            rr, cc = circle(blob[0],blob[1],radius,self._data.shape)
            tolabel[rr, cc] = 1

        labels, num_labels = label(tolabel)
        if num_labels != len(blobs):
            warnings.warn('Blobs have melded, fitting may be difficult',UserWarning)

        self._labels = labels

        return labels

    def plot_labels(self, withfits = False,diameter = None, **kwargs):
        '''
        Generate a plot of the found peaks, individually
        '''

        #check if the fitting has been performed yet, warn user if it hasn't
        if withfits:
            if self._fits is None:
                withfits = False
                warnings.warn('Blobs have not been fit yet, cannot show fits', UserWarning)
            else:
                fits = self._fits

        #pull the labels and the data from the object
        labels = self._labels
        data = self.data

        #check to see if data has been labelled
        if labels is None:
            labels = self.label_blobs(diameter=diameter)
            if labels is None:
                warnings.warn('Labels were not available', UserWarning)

                return None

        #find objects from labelled data
        my_objects = find_objects(labels)

        #generate a nice layout
        nb_labels = len(my_objects)

        nrows = int(np.ceil(np.sqrt(nb_labels)))
        ncols = int(np.ceil(nb_labels / nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

        for n, (obj, ax) in enumerate(zip(my_objects,axes.ravel())):
            ex = (obj[1].start,obj[1].stop-1,obj[0].stop-1,obj[0].start)
            ax.matshow(data[obj],extent=ex,**kwargs)
            if withfits:
                #generate the model fit to display, from parameters.
                dict_params = dict(fits.loc[n].dropna())

                #recenter
                dict_params['x0'] -= obj[1].start
                dict_params['y0'] -= obj[0].start

                params = Gauss2D.dict_to_params(dict_params)
                fake_data = Gauss2D.gen_model(data[obj], *params)
                ax.contour(fake_data, extent=ex,colors='w',origin='image')

        ## Remove empty plots
        for ax in axes.ravel():
            if not(len(ax.images)) and not(len(ax.lines)):
                fig.delaxes(ax)

        fig.tight_layout()

        #return the fig and axes handles to user for later manipulation.
        return fig, axes

    def filter_blobs(self,minamp=None,maxamp=None):
        amps = self.blobs[:,3]
        if maxamp is None:
            maxamp = amps.max()

        if maxamp is None:
            minamp = amps.min()

        self.blobs = self.blobs[np.logical_and(maxamp > amps, amps > minamp)]

        return self.blobs

    def fit_blobs(self, diameter = None, quiet = True):
        labels = self._labels
        data = self.data

        if labels is None:
            self.label_blobs(diameter=diameter)
            labels = self._labels
            if labels is None:
                warnings.warn('Labels were not available', UserWarning)
                return None

        my_objects = find_objects(labels)
        peakfits = pd.DataFrame(index=np.arange(len(my_objects)),\
         columns=['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset'], dtype=float)

        for i, obj in enumerate(my_objects):
            mypeak = Gauss2D(data[obj])
            mypeak.optimize_params_ls(modeltype = self.modeltype, quiet = quiet)
            fit_coefs = mypeak.opt_params_dict()

            #need to place the fit coefs in the right place
            fit_coefs['y0'] += obj[0].start
            fit_coefs['x0'] += obj[1].start

            peakfits.loc[i] = fit_coefs

        #we know that when the fit fails it returns 0, so replace with NaN
        peakfits.replace(0,np.NaN)
        peakfits[['sigma_x','sigma_y']] = np.abs(peakfits[['sigma_x','sigma_y']])
        self._fits = peakfits

        return peakfits


    def prune_blobs(self,diameter):
            """
            Pruner method takes blobs list with the third column replaced by
            intensity instead of sigma and then removes the less intense blob
            if its within diameter of a more intense blob.

            Adapted from _prune_blobs in skimage.feature.blob

            Parameters
            ----------
            blobs : ndarray
                A 2d array with each row representing 3 values, ``(y,x,intensity)``
                where ``(y,x)`` are coordinates of the blob and ``intensity`` is the
                intensity of the blob (value at (x,y)).
            diameter : float
                Allowed spacing between blobs

            Returns
            -------
            A : ndarray
                `array` with overlapping blobs removed.
            """

            #make a copy of blobs otherwise it will be changed
            myBlobs = self.blobs

            #cycle through all possible pairwise cominations of blobs
            for blob1, blob2 in itt.combinations(myBlobs, 2):
                #take the norm of the difference in positions and compare
                #with diameter
                if norm((blob1-blob2)[0:2]) < diameter:
                    #compare intensities and use the third column to keep track
                    #of which blobs to toss
                    if blob1[3] > blob2[3]:
                        blob2[3] = -1
                    else:
                        blob1[3] = -1

            # set internal blobs array to blobs_array[blobs_array[:, 2] > 0]
            self._blobs = np.array([a for b, a in zip(myBlobs,self.blobs) if b[3] > 0])

            #Return a copy of blobs incase user wants a onliner
            return self.blobs

    def remove_edge_blobs(self,distance):

        ymax, xmax = self._data.shape

        my_blobs = np.array([blob for blob in self.blobs if distance < blob[0] < ymax - distance\
                    and distance < blob[1] < xmax - distance])

        my_blobs = my_blobs[my_blobs[:,3].argsort()]

        self._blobs = my_blobs
        return my_blobs

    def plot_blobs(self, diameter = None, size = 12, **kwargs):

        if self.blobs is None:
            raise UserWarning('No blobs have been found')

        fig, ax = plt.subplots(1, 1,figsize=(size,size))

        ax.matshow(self.data,**kwargs)
        for blob in self.blobs:
            y, x, s, r = blob
            if diameter is None:
                diameter = s*4

            c = plt.Circle((x, y), radius=diameter/2, color='r', linewidth=1,\
                           fill=False)
            ax.add_patch(c)
            if self.data.dtype != float:
                r = int(r)
                fmtstr = '{}'
            else:
                fmtstr = '{:.3f}'

            ax.annotate(fmtstr.format(r), xy=(x, y),xytext = (x+diameter/2,y+diameter/2),\
                textcoords = 'data', color='k', backgroundcolor=(1,1,1,0.5) ,xycoords='data')

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
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach
    Examples
    --------
    >>> from skimage import data, feature
    >>> feature.blob_dog(data.coins(), threshold=.5, max_sigma=40)
    array([[  45.      ,  336.      ,   16.777216],
           [  52.      ,  155.      ,   16.777216],
           [  52.      ,  216.      ,   16.777216],
           [  54.      ,   42.      ,   16.777216],
           [  54.      ,  276.      ,   10.48576 ],
           [  58.      ,  100.      ,   10.48576 ],
           [ 120.      ,  272.      ,   16.777216],
           [ 124.      ,  337.      ,   10.48576 ],
           [ 125.      ,   45.      ,   16.777216],
           [ 125.      ,  208.      ,   10.48576 ],
           [ 127.      ,  102.      ,   10.48576 ],
           [ 128.      ,  154.      ,   10.48576 ],
           [ 185.      ,  347.      ,   16.777216],
           [ 193.      ,  213.      ,   16.777216],
           [ 194.      ,  277.      ,   16.777216],
           [ 195.      ,  102.      ,   16.777216],
           [ 196.      ,   43.      ,   10.48576 ],
           [ 198.      ,  155.      ,   10.48576 ],
           [ 260.      ,   46.      ,   16.777216],
           [ 261.      ,  173.      ,   16.777216],
           [ 263.      ,  245.      ,   16.777216],
           [ 263.      ,  302.      ,   16.777216],
           [ 267.      ,  115.      ,   10.48576 ],
           [ 267.      ,  359.      ,   16.777216]])
    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}sigma`.
    """
    assert_nD(image, 2)

    image = img_as_float(image)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(math.log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    #NOTE a faster gaussian_filter would significantly speed this operation
    #up.
    gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * sigma_list[i] for i in range(k)]
    image_cube = np.dstack(dog_images)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3, 3, 3)),
                                  threshold_rel=0.0,
                                  exclude_border=False)
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, 2] = sigma_list[local_maxima[:, 2]]
    local_maxima = lm
    return local_maxima

################################################################################
#                          Spectral Peak Finding Part                          #
################################################################################

class SpectralPeakFinder(object):
    '''
    A class used to find peaks in data that has one spatial and one spectral
    and one time dimension

    Data is assumed to have dimensions time (0), space (1), spectral (2)
    '''

    #NOTE that the way this class is implemented it does not hide any of its
    #variables or methods from the user.


    def __init__(self, data):
        '''
        A class designed to find peaks in spectral/spatial/time data
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('data is not a numpy array')

        #this is **VERY** data _un_aware!
        #this makes a copy, which means that original data should be safe
        #we're casting to a signed 32 bit int which has enough bit depth to accomodate
        #the original data (uint16) but also allows negative numbers.
        self.data = data.astype(int)

    def remove_background(self):
        '''
        Remove background from the data cube.

        This method uses a relatively simple algorithm that first takes the mean along the
        time dimension and then the median along the spatial dimension

        The assumption here is that peaks are relatively sparse along the spatial
        dimension

        NOTE: This function mutates the data internally
        '''
        #pull internal data
        data = self.data
        #take the median value along the time and spatial dimensions
        #keep the dimensions so that broadcasting will work properly

        #bg = np.median(data,axis=(0,1), keepdims=True)
        #this is much faster than the above but gives approximately the same results
        bg = np.median(data.mean(0),0)

        self.data = data-bg

    def fix_hot_pixels(self,cutoff=9):
        '''
        A method to remove "Salt and Pepper" noise from the image stack

        This method assumes that hot pixels do not vary much with time and uses
        this property to avoid performing a median filter for every time point.

        Remember this function mutates the data internally
        '''
        #pull internal data
        data = self.data

        #calc the _mean_ projection
        #the assumption is that if the hot pixel is in one frame it will be in all of them
        #and the whole point of this method is to only perform the median filter once
        mean_data =data.mean(0)

        #do the one median filter, use a 3x3 footprint
        #some articles suggest that a 2x2 is fine, but I'm not sure if I buy that
        #NOTE: that because we're filtering _single_ pixels
        mean_data_med = median_filter(mean_data,3)

        #subtract the median filtered data from the unfiltered data
        data_minus = mean_data-mean_data_med

        #calculate the z-score for each pixel
        z_score = (data_minus-data_minus.mean())/data_minus.std()

        #find the points to remove
        picked_points = (z_score>cutoff)*mean_data

        #remove them from the data
        data -= picked_points

        #return the number of points removed
        return np.count_nonzero(picked_points)



    def calc_FoM(self, width,s_lambda = 3, s_time = 3, use_max = False):
        '''
        Calculate the figure of merit (FoM) of a dataset (t, x, and lambda)

        In this case our figure of merit is calculated as the _maximum_ value
        along the spectral dimension aver the

        Parameters
        ----------
        data : ndarray (NxMxK)
            the array overwhich to calculate the SNR, assumes that it
            has dimensions (time, position, spectrum)
        width : int
            the width overwhich to calculate the average in the spatial dimention
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
        '''

        #before we make another copy we should trash the old one, if it exists
        #if we don't do this it can lead to a memory leak.
        try:
            del self.g_mean_data
        except AttributeError as e:
            pass

        #First calculate the moving average of the data along the spatial dimension
        #cast as float64 for better precision, this is necessary for the later
        #gaussian filters, but might as well do it now to avoid making more copies
        #of the data than necessary.

        if use_max:
            data = self.data.max(0,keepdims=True).astype(float)
        else:
            data = self.data.astype(float)

        mean_data = uniform_filter1d(data,width,axis=1)

        #calculate the gaussian blue along the spectral and time dimensions
        if s_time == 0 and s_lambda == 0:
            g_mean_data = mean_data
        else:
            g_mean_data = gaussian_filter(mean_data,(s_time,0,s_lambda))

        g_mean_data_mean = g_mean_data.mean(axis=(0,2))
        g_mean_data_std = g_mean_data.std(axis=(0,2))
        g_mean_data_max = g_mean_data.max(axis=(0,2))

        FoM = (g_mean_data_max-g_mean_data_mean)/g_mean_data_std

        self.FoM =  FoM
        self.g_mean_data = g_mean_data

    def find_peaks(self,width,cutoff=7, cutoff_high=np.inf, show=False):
        '''
        A function that finds peaks in the FoM trace.
        '''

        FoM = self.FoM
        g_mean_data = self.g_mean_data

        #find the local maxima in the SNR trace
        #presmooth might make sense here
        peaks = argrelmax(FoM*(FoM > cutoff),order=width)[0]

        peaks = peaks[FoM[peaks] < cutoff_high]

        #Show the peaks?
        if show:
            fig, ax = plt.subplots(1,1)
            ax.plot(FoM)
            ax.plot(peaks,FoM[peaks],'ro')
            ax.axis('tight')

        self.peaks = peaks

    def refine_peaks(self, window_width=8):
        '''
        A function that refines peaks.

        Because of the way the FoM is calculated the highest SNR region isn't
        identified because the noise is approximated by the std. This function
        will search the nearby are for a peak (using the smoothed data) and will
        return that point instead.

        Parameters
        ----------
        window_width : int (optional)
            the window in which to search for a peak.
        '''
        new_peaks = []

        #take the max of the data along the time axis
        max_data = self.g_mean_data.max(0)
        ny, nx = max_data.shape

        ny = window_width*2

        #NOTE: this implementation is pretty slow. But I'm not quite sure how
        #to speed it up.
        for peak in self.peaks:
            #find the max
            dy,dx = np.unravel_index(max_data[peak-window_width:peak+window_width].argmax(),(ny,nx))
            new_peaks.append(peak-window_width+dy)

        self.peaks = np.array(new_peaks)

    def _plot_peaks_lines(self):
        '''
        A helper function to plot a max intensity projection with redlines marking
        the location of the found peaks.
        '''
        figmat, axmat = plt.subplots(1,1,squeeze=True,sharex = True)
        axmat.matshow(self.data.max(0))
        axmat.set_yticks(self.peaks)
        for peak in self.peaks:
            axmat.axhline(peak,color='r')

    def plot_peaks(self):
        '''
        A utility function to plot the found peaks.
        '''

        peaks = self.peaks
        FoM = self.FoM
        g_mean_data = self.g_mean_data
        nz, ny, nx = g_mean_data.shape
        #plot the found peaks in the SNR trace

        print( g_mean_data.shape)

        #self._plot_peaks_lines()

        for peak in peaks:

            #need to ensure a reasonable ratio
            ratio = nz/nx
            if ratio < 0.05:
                ratio = 0.05

            fig,ax = plt.subplots(2,1,squeeze=True,sharex = True,figsize = (12,12*ratio*2))

            ax[0].matshow(g_mean_data[:,peak,:])
            ax[0].axis('tight')
            ax[0].set_xticks([])

            ax[1].plot(g_mean_data[:,peak,:].max(0))
            ax[1].axis('tight')

            fig.suptitle('{}, Max SNR {:.3f}'.format(peak, FoM[peak]),y=1,fontsize=14)

            fig.tight_layout()

class SpectralPeakFinder1d(SpectralPeakFinder):
    '''
    A class to find peaks in a single frame.
    '''

    def __init__(self, data):
        #reshape the data so that it can use the previous methods without changes
        super().__init__(data.reshape(1,*data.shape))

    #overload the plot peaks function
    def plot_peaks(self):
        '''
        A utility function to plot the found peaks.
        '''

        peaks = self.peaks
        FoM = self.FoM
        g_mean_data = self.g_mean_data
        nz, ny, nx = g_mean_data.shape
        #plot the found peaks in the SNR trace

        self._plot_peaks_lines()

        for peak in peaks:
            fig,ax = plt.subplots(1,1,squeeze=True,sharex = True)

            ax.plot(g_mean_data[0,peak,:])
            ax.axis('tight')
            ax.set_title('{}, Max SNR {:.3f}'.format(peak, FoM[peak]),y=1,fontsize=14)
