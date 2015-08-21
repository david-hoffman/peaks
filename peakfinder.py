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

#we need ittertools for the pruner function defined below
import itertools as itt

#we need a few extra features from matplot lib
import matplotlib.pyplot as plt
from matplotlib.path import Path #Needed to create shapes
import matplotlib.patches as patches #needed so show shapes on top of graphs

#We want to be able to warn the user about potential problems
import warnings


from scipy.ndimage.measurements import label, find_objects

#import our 2D gaussian fitting class
from .gauss2d import Gauss2D

#need pandas for better data containers
import pandas as pd

from numpy.linalg import norm

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

        if value.shape[1] != 4:
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
            thresh = kwargs['threshold']
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
            blobs = blob_dog(scaled_data,**kwargs)
        else:
            blobs = None

        #if no peaks found alert the user, but don't break their program
        if blobs is None or len(blobs) == 0:
            warnings.warn('No peaks found',UserWarning)

        else:
            #blobs, as returned, has the third index as the estimated width
            #for our application it will be beneficial to have the intensity at
            #the estimated center as well
            blobs = np.array([[i[0], i[1], i[2], self.data[i[0],i[1]]] for i in blobs])

            #sort blobs by the max am value
            blobs[blobs[:,3].argsort()]

        self._blobs = blobs
        return blobs

    def label_blobs(self, diameter = None):
        '''
        This function will create a labeled image from blobs
        essentially it will be circles at each location with diamter of
        4 sigma
        '''

        tolabel = np.zeros_like(self.data)
        blobs = self.blobs

        if blobs is None:
            #try to find blobs
            self.find_blobs()
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

    def plot_labels(self, withfits = False, **kwargs):
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
            self.label_blobs()
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

        for obj, ax in zip(my_objects,axes.ravel()):
            ax.matshow(data[obj],extent=(obj[1].start,obj[1].stop-1,obj[0].stop-1,obj[0].start),**kwargs)
            if withfits:
                #generate the model fit to display, from parameters.
                ax.contour(Gauss2D.gen_model(data[obj], *self.fit_to_params(fits,n)),\
                extent=(obj[1].start,obj[1].stop-1,obj[0].stop-1,obj[0].start),colors='w')

        ## Remove empty plots
        for ax in axes.ravel():
            if not(len(ax.images)) and not(len(ax.lines)):
                fig.delaxes(ax)

        fig.tight_layout()

        #return the fig and axes handles to user for later manipulation.
        return fig, axes

    def fit_to_params(self, fit, idx):
        my_dict = dict(fit.loc[idx].dropna())

        num_params = len(my_dict)

        keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset']

        #adjust the dictionary size
        if num_params < 7:
            keys.remove('rho')

        if num_params < 6:
            keys.remove('sigma_y')

        return [my_dict[k] for k in keys]


    def fit_blobs(self, diameter = None, quiet = True):
        labels = self._labels
        data = self.data

        if labels is None:
            self.label_blobs()
            if labels is None:
                warnings.warn('Labels were not available', UserWarning)
                return None

        my_objects = find_objects(labels)
        peakfits = pd.DataFrame(index=np.arange(len(my_objects)),\
         columns=['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset'])

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
        peakfits[['sigma_x','sigma_y']] = abs(peakfits[['sigma_x','sigma_y']])
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

    def plot_blobs(self, diameter = None, **kwargs):

        if self.blobs is None:
            raise UserWarning('No blobs have been found')

        fig, ax = plt.subplots(1, 1,figsize=(12,12))

        ax.matshow(self.data,**kwargs)
        for blob in self.blobs:
            y, x, s, r = blob
            if diameter is None:
                diameter = s*4

            c = plt.Circle((x, y), radius=diameter/2, color='r', linewidth=1,\
                           fill=False)
            ax.add_patch(c)
            ax.annotate('{:.3f}'.format(self.data[y,x]), xy=(x, y),xytext = (x+diameter/2,y+diameter/2),\
                textcoords = 'data', color='k', backgroundcolor=(1,1,1,0.5) ,xycoords='data')

        return fig, ax
