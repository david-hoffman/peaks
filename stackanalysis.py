'''
A set of classes for analyzing data stacks that contain punctate data
'''
import numpy as np
import pandas as pd
from .gauss2d import Gauss2D
from .peakfinder import PeakFinder

class StackAnalyzer(object):
    """
    A parent class for more specialized analysis classes
    """
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def findpeaks(self):
        '''
        A method to find peaks, should have data passed into it, that way child
        classes can decide how to find peaks initially.
        '''
        raise NotImplementedError

    def sliceMaker(self, y0, x0, width):
        '''
        A utility function to generate slices for later use
        '''

        #pull stack from object
        stack = self.stack

        #calculate max extents
        ymax = self.stack.shape[1]
        xmax = self.stack.shape[2]

        #calculate the start and end
        half1 = width//2
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

        #return a list of slices
        return [slice(ystart,yend), slice(xstart, xend)]

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

        '''

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

            #try to update the y0/x0 values
            #if the fit has failed, these will be nan and the operation will raise a value error
            #doing nothing leaves the values intact

            #make the slice
            myslice = self.sliceMaker(y0, x0, width)

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

            fit.optimize_params_ls(popt_d, **kwargs)

            #if there was an error performing the fit, try again without a guess
            if fit.error:
                fit.optimize_params_ls(modeltype = modeltype, **kwargs)

            #if there's still an error move on to the next fit
            if not fit.error:
                popt_d = fit.opt_params_dict()
                popt_d['x0']+=xstart
                popt_d['y0']+=ystart

                popt_d['slice']=s

                toreturn.append(popt_d.copy())

                y0 = int(round(popt_d['y0']))
                x0 = int(round(popt_d['x0']))
            else:
                bad_fit = fit.opt_params_dict()
                bad_fit['slice']=s

                toreturn.append(bad_fit.copy())

        return toreturn


class PSFStackAnalyzer(StackAnalyzer):
    """
    docstring for PSFStackAnalyser
    """

    def __init__(self, stack, psfwidth = 1.68, **kwargs):
        super().__init__(stack)
        self.psfwidth = psfwidth
        self.peakfinder = PeakFinder(self.stack.max(0),self.psfwidth,**kwargs)
        self.peakfinder.find_blobs()

    def fitPeaks(self, fitwidth):
        '''
        Fit all peaks found by peak finder
        '''

        blobs = self.peakfinder.blobs

        fits = []

        for blob in blobs:
            y,x,w,amp = blob

            myslice = self.sliceMaker(y,x,fitwidth)

            ystart = myslice[0].start
            xstart = myslice[1].start

            #insert the equivalent of `:` at the beginning
            myslice.insert(0,slice(None, None, None))

            substack = self.stack[myslice]

            my_max = np.unravel_index(substack.argmax(),substack.shape)

            myslice[0] = my_max[0]
            substack = self.stack[myslice]

            #prep our container
            peakfits = []

            #initial fit
            max_z = Gauss2D(substack)
            max_z.optimize_params_ls()

            #recenter the coordinates and add a slice varaible
            opt_params = max_z.opt_params_dict()
            opt_params['slice']=my_max[0]
            opt_params['x0']+=xstart
            opt_params['y0']+=ystart

            #append to our list
            peakfits.append(opt_params.copy())

            #pop the slice parameters
            opt_params.pop('slice')

            forwardrange = range(my_max[0]+1,self.stack.shape[0])
            backwardrange = reversed(range(0, my_max[0]))

            peakfits+=self.fitPeak(forwardrange, fitwidth, opt_params.copy(), quiet = True)
            peakfits+=self.fitPeak(backwardrange, fitwidth, opt_params.copy(), quiet = True)

            peakfits_df = pd.DataFrame(peakfits)

            fits.append(peakfits_df.set_index('slice').sort())

        self.fits = fits

        return fits


class SIMStackAnalyzer(StackAnalyzer):
    """
    docstring for SIMStackAnalyser
    """
    def __init__(self, norients, nphases, **kwargs):
        super().__init__(**kwargs)
        self.arg = arg
