'''
A set of classes for analyzing data stacks that contain punctate data
'''

class StackAnalyzer(object):
    """docstring for StackAnalyzer"""
    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def findpeaks(self):
        '''
        A method to find peaks, should have data passed into it, that way child
        classes can decide how to find peaks initially.
        '''
        raise NotImplementedError

    def sliceMaker(cls, y0, x0, width):
        '''
        A utility function to generate slices for later use
        '''

        stack = self.stack
        ymax = self.stack.shape[1]
        xmax = self.stack.shape[2]

        ystart = y0-width//2
        xstart = x0-width//2

        yend = ystart+width
        xend = ystart+width

        if ystart < 0:
            ystart = 0
        if xstart < 0 :
            xstart = 0

        if yend > ymax:
            yend = ymax
        if xend > xmax:
            xend = xmax

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

        stack : ndarray (3 dimensions)

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

        for slic in slices:

            #try to update the y0/x0 values
            #if the fit has failed, these will be nan and the operation will raise a value error
            #doing nothing leaves the values intact

            #make the slice
            myslice = sliceMaker(y0, x0, width, stack.shape[1],stack.shape[2])

            #pull the starting values from it
            ystart = myslice[0].start
            xstart = myslice[1].start

            #insert the z-slice number
            myslice.insert(0,slic)

            #set up the fit and perform it using last best params
            fit = Gauss2D((PSF[myslice]))

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

                popt_d['slice']=slic

                toreturn.append(popt_d.copy())

                y0 = int(round(popt_d['y0']))
                x0 = int(round(popt_d['x0']))
            else:
                bad_fit = fit.opt_params_dict()
                bad_fit['slice']=slic

                toreturn.append(bad_fit.copy())

        return toreturn


class PSFStackAnalyser(StackAnalyzer):
    """
    docstring for PSFStackAnalyser
    """

    def __init__(self, arg, **kwargs):
        super().__init__(**kwargs)
        self.arg = arg

class SIMStackAnalyser(StackAnalyzer):
    """
    docstring for SIMStackAnalyser
    """
    def __init__(self, norients, nphases, **kwargs):
        super().__init__(**kwargs)
        self.arg = arg
