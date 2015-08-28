'''
A set of classes for analyzing data stacks that contain punctate data
'''
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.mlab import griddata
from .gauss2d import Gauss2D
from .peakfinder import PeakFinder

#TODO
#Need to move all the fitting stuff into its own class and abstract as much
#functionality from gauss2d into a parent class that can be subclassed for
#each type of peak. Hopefully regardless of dimensionality.

#theres not a huge speed up with more than ~4 cores, this might be because each
#process needs its own copy of `stack`, or the communication overhead quickly
#outweighs the higher processing power.

def gauss_fit(xdata, ydata, withoffset = True,trim = None, guess_z = None):

    def nmoment(x, counts, c, n):
        '''
        A helper function to calculate moments of histograms
        '''
        return np.sum((x-c)**n*counts) / np.sum(counts)

    def gauss_no_offset(x, amp, x0, sigma_x):
        '''
        Helper function to fit 1D Gaussians
        '''

        return amp*np.exp(-(x-x0)**2/(2*sigma_x**2))

    def gauss(x, amp, x0, sigma_x, offset):
        '''
        Helper function to fit 1D Gaussians
        '''

        return amp*np.exp(-(x-x0)**2/(2*sigma_x**2))+offset

    offset = ydata.min()
    ydata_corr = ydata-offset

    if guess_z is None:
        x0 = nmoment(xdata,ydata_corr,0,1)
    else:
        x0 = guess_z

    sigma_x = np.sqrt(nmoment(xdata,ydata_corr,x0,2))

    p0 = np.array([ydata_corr.max(),x0,sigma_x, offset])

    if trim is not None:
        args =  abs(xdata-x0) < trim*sigma_x
        xdata=xdata[args]
        ydata=ydata[args]

    try:
        if withoffset:
            popt, pcov = curve_fit(gauss, xdata, ydata, p0=p0)
        else:
            popt, pcov = curve_fit(gauss_no_offset, xdata, ydata, p0=p0[:3])
            popt = np.insert(popt,3,offset)
    except RuntimeError as e:
        popt=p0*np.nan

    return popt

def grid(x, y, z, resX=1000, resY=1000):
    "Convert 3 column data to matplotlib grid"
    if not np.isfinite(x+y+z).all():
        raise ValueError('x y or z is not finite')

    xi = np.linspace(x.min(), x.max(), resX)
    yi = np.linspace(y.min(), y.max(), resY)
    Z = griddata(x, y, z, xi, yi,interp='linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

def scatterplot(z, y, x,cmap = 'gnuplot2'):
    '''
    A way to make a nice scatterplot with contours.
    '''
    mymax = z.max()
    mymin = z.min()
    fig, ax = plt.subplots(1,1, squeeze=True, figsize= (6,6))
    X, Y, Z = grid(x,y,z)
    conts=20
    s= ax.contourf(X, Y, Z,conts,origin='upper',cmap = cmap)
    ax.contour(X, Y, Z,conts,colors='k',origin='upper')
    ax.scatter(x,y,c='c')
    ax.invert_yaxis()
    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(s,cax=color_axis)
    return fig, ax

class StackAnalyzer(object):
    """
    A parent class for more specialized analysis classes
    """
    def __init__(self, stack):
        super().__init__()
        #stack is the image stack to be analyzed
        self.stack = stack

    def findpeaks(self):
        '''
        A method to find peaks, should have data passed into it, that way child
        classes can decide how to find peaks initially.
        '''
        raise NotImplementedError

    def sliceMaker(self, y0, x0, width):
        '''
        A utility function to generate slices for later use.

        Parameters
        ----------
        y0 : int
            center y position of the slice
        x0 : int
            center x position of the slice
        width : int
            Width of the slice

        Returns
        -------
        slices : list
            A list of slice objects, the first one is for the y dimension and
            and the second is for the x dimension.

        Notes
        -----
        The method will automatically coerce slices into acceptable bounds.
        '''

        #pull stack from object
        stack = self.stack

        #calculate max extents
        zmax, ymax, xmax = self.stack.shape

        #calculate the start and end
        half1 = width//2
        #we need two halves for uneven widths
        half2 = width-half1
        ystart = y0 - half1
        xstart = x0 - half1
        yend = y0 + half2
        xend = x0 + half2

        #coerce values into an acceptable range
        if ystart < 0:
            ystart = 0
        if xstart < 0 :
            xstart = 0

        if yend >= ymax:
            yend = ymax - 1
        if xend >= xmax:
            xend = xmax - 1

        toreturn = [slice(ystart,yend), slice(xstart, xend)]

        #check to see if we've made valid slices, if not raise an error
        if ystart >= yend or xstart >= xend:
            raise RuntimeError('sliceMaker made a zero length slice '+repr(toreturn))
        #return a list of slices
        return toreturn

    def fitPeak(self, slices, width, startingfit, **kwargs):
        '''
        Method to fit a peak through the stack.

        The method will track the peak through the stack, assuming that moves are relatively small
        from one slice to the next

        Parameters
        ----------
        slices : iterator
            an iterator which dictates which slices to fit, should yeild integers only

        width : integer
            width of fitting window

        startingfit : dict
            fit coefficients

        Returns
        -------
        list : list of dicts
            A list of dictionaries containing the best fits. Easy to turn into
            a DataFrame

        '''

        #pull stack
        stack = self.stack

        #set up our variable to return
        toreturn = []

        #grab the starting fit parameters
        popt_d = startingfit.copy()

        y0 = int(round(popt_d['y0']))
        x0 = int(round(popt_d['x0']))

        if len(popt_d) == 6:
            modeltype = 'norot'
        elif len(popt_d) == 5:
            modeltype = 'sym'
        else:
            modeltype = 'full'

        for s in slices:

            #make the slice
            try:
                myslice = self.sliceMaker(y0, x0, width)
            except RuntimeError as e:
                print('Fit window moved to edge of ROI')
                break
            else:
                #pull the starting values from it
                ystart = myslice[0].start
                xstart = myslice[1].start

                #insert the z-slice number
                myslice.insert(0,s)

                #set up the fit and perform it using last best params
                fit = Gauss2D(stack[myslice])

                #move our guess coefs back into the window
                popt_d['x0']-=xstart
                popt_d['y0']-=ystart
                #leave this in for now for easier debugging in future.
                try:
                    fit.optimize_params_ls(popt_d, **kwargs)
                except TypeError as e:
                    print(repr(myslice))
                    raise e

                #if there was an error performing the fit, try again without a guess
                if fit.error:
                    fit.optimize_params_ls(modeltype = modeltype, **kwargs)

                #if there's not an error update center of fitting window and move
                #on to the next fit
                if not fit.error:
                    popt_d = fit.opt_params_dict()
                    popt_d['x0']+=xstart
                    popt_d['y0']+=ystart

                    popt_d['slice']=s

                    toreturn.append(popt_d.copy())

                    y0 = int(round(popt_d['y0']))
                    x0 = int(round(popt_d['x0']))
                else:
                    #if the fit fails, make sure to _not_ update positions.
                    bad_fit = fit.opt_params_dict()
                    bad_fit['slice']=s

                    toreturn.append(bad_fit.copy())

        return toreturn

class PSFStackAnalyzer(StackAnalyzer):
    """
    A specialized version of StackAnalyzer for PSF stacks.
    """

    def __init__(self, stack, psfwidth = 1.68, **kwargs):
        super().__init__(stack)
        self.psfwidth = psfwidth

        self.peakfinder = PeakFinder(median_filter(self.stack.max(0),3),self.psfwidth,**kwargs)

        self.peakfinder.find_blobs()
        #should have a high accuracy mode that filters the data first and finds
        #the slice with the max value before finding peaks.


    def _fitPeaks_sub(self, fitwidth, blob, **kwargs):
        y,x,w,amp = blob

        myslice = self.sliceMaker(y,x,fitwidth)

        ystart = myslice[0].start
        xstart = myslice[1].start

        #insert the equivalent of `:` at the beginning
        myslice.insert(0,slice(None, None, None))

        substack = self.stack[myslice]

        #we could do median filtering on the substack before attempting to
        #find the max slice!

        #this could still get messed up by salt and pepper noise.
        #my_max = np.unravel_index(substack.argmax(),substack.shape)
        #use the sum of each z-slice
        my_max = substack.sum((1,2)).argmax()

        #now change my slice to be that zslice
        myslice[0] = my_max
        substack = self.stack[myslice]

        #prep our container
        peakfits = []

        #initial fit
        max_z = Gauss2D(substack)
        max_z.optimize_params_ls(**kwargs)

        if np.isfinite(max_z.opt_params).all():

            #recenter the coordinates and add a slice variable
            opt_params = max_z.opt_params_dict()
            opt_params['slice']=my_max
            opt_params['x0']+=xstart
            opt_params['y0']+=ystart

            #append to our list
            peakfits.append(opt_params.copy())

            #pop the slice parameters
            opt_params.pop('slice')

            forwardrange = range(my_max+1,self.stack.shape[0])
            backwardrange = reversed(range(0, my_max))

            peakfits+=self.fitPeak(forwardrange, fitwidth, opt_params.copy(), quiet = True)
            peakfits+=self.fitPeak(backwardrange, fitwidth, opt_params.copy(), quiet = True)

            #turn everything into a data frame for easy manipulation.
            peakfits_df = pd.DataFrame(peakfits)
            #convert sigmas to positive values
            peakfits_df[['sigma_x','sigma_y']] = abs(peakfits_df[['sigma_x','sigma_y']])

            return peakfits_df.set_index('slice').sort()
        else:
            print('blob {} is unfittable'.format(blob))
            return None

    def fitPeaks(self, fitwidth, nproc = 1, **kwargs):
        '''
        Fit all peaks found by peak finder

        Parameters
        ----------
        fitwidth : int
            Sets the size of the fitting window

        Returns
        -------
        list : list of DataFrames
            A list of DataFrames with each DataFrame holding the fits of one peak
        '''

        blobs = self.peakfinder.blobs

        if nproc == 1:
            fits = [self._fitPeaks_sub(fitwidth, blob, **kwargs) for blob in blobs]
        elif 1 < nproc <= os.cpu_count():
            #multiprocessing version, order **NOT** retained
            print('multiprocessing engaged with {} cores'.format(nproc))
            with mp.Pool(nproc) as p:
                fits = [p.apply_async(self._fitPeaks_sub, args=(fitwidth, blob), kwds =kwargs) for blob in blobs]
                fits = [pp.get() for pp in fits]
        else:
            raise ValueError('nproc = {} while the maximum number of cores is {}'.format(nproc, os.cpu_count()))
        #clear nones
        fits = [fit for fit in fits if fit is not None]

        self.fits = fits

        return fits

    def calc_psf_params(self,subrange = slice(None,None,None),**kwargs):
        fits = self.fits

        psf_params = []

        for fit in fits:
            #first fit the amplitudes
            #pull values from DataFrame
            tempfit = fit.dropna().loc[subrange]
            z = tempfit.index.values
            #TODO
            #need to make this robust to different fitting models.
            amp, x, y, s_x, s_y =  tempfit[['amp', 'x0', 'y0', 'sigma_x', 'sigma_y']].values.T
            popt = gauss_fit(z,amp,**kwargs)
            if np.isfinite(popt).all():
                famp, z0, sigma_z, offset = popt
                x0 = np.interp(z0,z,x)
                y0 = np.interp(z0,z,y)
                sigma_x = np.interp(z0,z,s_x)
                sigma_y = np.interp(z0,z,s_y)

                psf_params.append({'z0' : z0, 'y0' : y0, 'x0' : x0, 'sigma_z' : abs(sigma_z), 'sigma_y' : sigma_y, 'sigma_x' : sigma_x, 'SNR' : famp/offset})

        self.psf_params = pd.DataFrame(psf_params)

    def plot_psf_params(self, feature):
        psf_params = self.psf_params
        fig, ax = scatterplot(psf_params[feature].values, psf_params.y0.values, psf_params.x0.values)
        fig.suptitle(feature)

class SIMStackAnalyzer(StackAnalyzer):
    """
    docstring for SIMStackAnalyser
    """
    def __init__(self, norients, nphases, **kwargs):
        super().__init__(**kwargs)
        self.arg = arg
