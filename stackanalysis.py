'''
A set of classes for analyzing data stacks that contain punctate data
'''
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
from scipy import ndimage as ndi
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from .gauss2d import Gauss2D
from .peakfinder import PeakFinder
from scipy.fftpack import fft

#TODO
#Need to move all the fitting stuff into its own class and abstract as much
#functionality from gauss2d into a parent class that can be subclassed for
#each type of peak. Hopefully regardless of dimensionality.

#Need to figure out a better way to multiprocess, maybe it makes more sense to
#to figure out how to send each stackanalyzer to each core.

#Isaac suggests trying to use Mr. Job

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
            popt = np.insert(popt, 3, offset)
    except RuntimeError as e:
        popt=p0*np.nan

    return popt

def grid(x, y, z, resX=1000, resY=1000, method = 'cubic'):
    "Convert 3 column data to matplotlib grid"
    if not np.isfinite(x+y+z).all():
        raise ValueError('x y or z is not finite')

    xi = np.linspace(x.min(), x.max(), resX)
    yi = np.linspace(y.min(), y.max(), resY)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y),method=method)
    return X, Y, Z

def scatterplot(z, y, x, ax = None, fig = None, cmap = 'gnuplot2',**kwargs):
    '''
    A way to make a nice scatterplot with contours.
    '''

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, squeeze =True, figsize =(6, 6))

    #split out the key words for the grid method
    #so that the rest can be passed onto the first contourf call.
    grid_kwargs = {}
    for k in ('resX', 'resY', 'method'):
        try:
            grid_kwargs[k] = kwargs.pop(k)
        except KeyError:
            pass

    X, Y, Z = grid(x, y, z, **grid_kwargs)

    mymax = np.nanmax(z)
    mymin = np.nanmin(z)

    #conts=np.linspace(mymin,mymax,20,endpoint=True)
    conts1=np.linspace(mymin, mymax, 30)
    conts2=np.linspace(mymin, mymax, 10)
    s= ax.contourf(X, Y, Z, conts1, origin='upper', cmap = cmap, zorder=0, **kwargs)
    ax.contour(X, Y, Z, conts2, colors='k', origin='upper', zorder=1)

    #if there's more than 100 beads to fit then don't make spots
    if len(x) > 100:
        mark = '.'
    else:
        mark = 'o'

    ax.scatter(x, y, c='c', zorder=2, marker =mark)
    ax.invert_yaxis()
    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(s, cax=color_axis)
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
                sub_stack = stack[myslice]
                fit = Gauss2D(sub_stack)

                #move our guess coefs back into the window
                popt_d['x0'] -= xstart
                popt_d['y0'] -= ystart
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
                    popt_d['x0'] += xstart
                    popt_d['y0'] += ystart

                    popt_d['slice'] = s
                    #calculate the apparent noise as the standard deviation
                    #of what's the residuals of the fit
                    popt_d['noise'] = (sub_stack-fit.fit_model).std()
                    toreturn.append(popt_d.copy())

                    y0 = int(round(popt_d['y0']))
                    x0 = int(round(popt_d['x0']))
                else:
                    #if the fit fails, make sure to _not_ update positions.
                    bad_fit = fit.opt_params_dict()
                    bad_fit['slice']=s
                    #noise of a failed fit is not really useful
                    popt_d['noise']= np.nan

                    toreturn.append(bad_fit.copy())

        return toreturn

    def fitPeaks(self, fitwidth, nproc =1, **kwargs):
        '''
        Fit all peaks found by peak finder, has the ability to split the peaks
        among multiple processors

        Parameters
        ----------
        fitwidth : int
            Sets the size of the fitting window
        nproc : int
            number of processors to use

        Returns
        -------
        list : list of DataFrames
            A list of DataFrames with each DataFrame holding the fits of one peak
        '''

        blobs = self.peakfinder.blobs

        if nproc > os.cpu_count():
            nproc = os.cpu_count()

        if nproc == 1:
            fits = [self._fitPeaks_sub(fitwidth, blob, **kwargs) for blob in blobs]
        elif 1 < nproc:
            #multiprocessing version, order of blobs **NOT** retained.
            #Error here if stack is larger than 1536x1536x126
            print('multiprocessing engaged with {} cores'.format(nproc))
            with mp.Pool(nproc) as p:
                par_func = self._fitPeaks_sub
                results = [p.apply_async(par_func, args=(fitwidth, blob), kwds =kwargs) for blob in blobs]
                fits = [pp.get() for pp in results]

        #clear nones (i.e. unsuccessful fits)
        fits = [fit for fit in fits if fit is not None]

        self.fits = fits

        return fits

class PSFStackAnalyzer(StackAnalyzer):
    """
    A specialized version of StackAnalyzer for PSF stacks.
    """

    def __init__(self, stack, psfwidth =1.68, **kwargs):
        super().__init__(stack)
        self.psfwidth = psfwidth
        #median filter to remove spikes
        self.peakfinder = PeakFinder(median_filter(self.stack.max(0), 3), self.psfwidth, **kwargs)
        self.peakfinder.find_blobs()
        #should have a high accuracy mode that filters the data first and finds
        #the slice with the max value before finding peaks.

    def _fitPeaks_sub(self, fitwidth, blob, **kwargs):
        y, x, w, amp = blob

        myslice = self.sliceMaker(y, x, fitwidth)

        ystart = myslice[0].start
        xstart = myslice[1].start

        #insert the equivalent of `:` at the beginning
        myslice.insert(0, slice(None, None, None))

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
            opt_params['slice'] = my_max
            opt_params['x0'] += xstart
            opt_params['y0'] += ystart

            #append to our list
            peakfits.append(opt_params.copy())

            #pop the slice parameters
            opt_params.pop('slice')

            forwardrange = range(my_max+1, self.stack.shape[0])
            backwardrange = reversed(range(0, my_max))

            peakfits += self.fitPeak(forwardrange, fitwidth, opt_params.copy(), quiet = True)
            peakfits += self.fitPeak(backwardrange, fitwidth, opt_params.copy(), quiet = True)

            #turn everything into a data frame for easy manipulation.
            peakfits_df = pd.DataFrame(peakfits)
            #convert sigmas to positive values
            peakfits_df[['sigma_x','sigma_y']] = abs(peakfits_df[['sigma_x','sigma_y']])

            return peakfits_df.set_index('slice').sort_index()
        else:
            print('blob {} is unfittable'.format(blob))
            return None

    def calc_psf_params(self, subrange = slice(None, None, None), **kwargs):
        fits = self.fits

        psf_params = []

        for fit in fits:
            #pull values from DataFrame
            tempfit = fit.dropna().loc[subrange]
            z = tempfit.index.values
            amp, x, y, s_x, s_y =  tempfit[['amp', 'x0', 'y0', 'sigma_x', 'sigma_y']].values.T

            #TODO
            #need to make this robust to different fitting models.

            #do the fit to a gaussian
            popt = gauss_fit(z, amp, **kwargs)

            #if the fit has not failed proceed
            if np.isfinite(popt).all():
                #pull fit parameters
                famp, z0, sigma_z, offset = popt

                #interpolate other values (linear only)
                x0 = np.interp(z0, z, x)
                y0 = np.interp(z0, z, y)
                sigma_x = np.interp(z0, z, s_x)
                sigma_y = np.interp(z0, z, s_y)

                noise = tempfit.noise.mean()
                #form a dictionary for easy DataFrame creation.
                psf_params.append({
                    'amp' : famp,
                    'z0' : z0,
                    'y0' : y0,
                    'x0' : x0,
                    'sigma_z' : abs(sigma_z),
                    'sigma_y' : sigma_y,
                    'sigma_x' : sigma_x,
                    'SNR' : famp/noise
                })

        #make the DataFrame and set it as a object attribute
        self.psf_params = pd.DataFrame(psf_params)

    def plot_psf_params(self, feature = 'z0',**kwargs):
        psf_params = self.psf_params
        fig, ax = scatterplot(psf_params[feature].values, psf_params.y0.values, psf_params.x0.values,**kwargs)
        ax.set_title(feature)
        return fig, ax

class SIMStackAnalyzer(StackAnalyzer):
    """
    docstring for SIMStackAnalyser
    """
    def __init__(self, stack, norients, nphases, psfwidth = 1.68, periods = 1, **kwargs):
        #make sure the stack has the right shape
        my_shape = stack.shape
        assert len(my_shape) == 3, "Stack has wrong number of dimensions"
        assert stack.shape[0] == norients*nphases, "Number of images does not equal orients*phases"

        super().__init__(stack)

        self.psfwidth = psfwidth
        self.nphases = nphases
        self.norients = norients
        self.periods = periods

        self.peakfinder = PeakFinder(median_filter(self.stack.max(0),3),self.psfwidth,**kwargs)
        self.peakfinder.find_blobs()
        #should have a high accuracy mode that filters the data first and finds
        #the slice with the max value before finding peaks.

    def sum_peaks(self,width):
        '''
        Find peaks, then sum area around them for whole stack.

        If we're going to do this _properly_ we need a way to find areas that
        _don't_ have any beads nearby inorder to calculate noise and offset.
        '''
        #fit the blobs first to find valid spots
        my_peaks = self.peakfinder

        peakfits = my_peaks.fit_blobs(diameter = width)
        #now reset the blobs to the fit values
        my_peaks.blobs = peakfits[['y0','x0','sigma_x','amp']].values

        #label again
        my_labels = my_peaks.label_blobs(diameter=width)

        #find all the objects.
        my_objects = ndi.find_objects(my_labels)

        my_medians = median(self.data,axis=(1,2))

        my_sums = array([self.data[:,obj[0],obj[1]].sum((1,2)) for obj in my_objects])

        self.sums = my_sums-my_medians
        #reset blobs to original
        self.peakfinder.find_blobs()

    def _fitPeaks_sub(self, fitwidth, blob, quiet=True, **kwargs):
        '''
        A sub function that can be dispatched to multiple cores for processing

        This function is specific to analyzing SIM data and is designed to fit
        substacks _without_ moving the fit window (i.e. it is assumed that drift
        is minimal).

        Parameters
        ----------
        fitwidth : int
            size of fitting window
        blob : list [int]
            a blob as returned by the find peak function

        Returns
        -------
        df : DataFrame
            A pandas DataFrame that contains all the fit parameters for a full
            stack.
        '''
        #pull parameters from the blob
        y,x,w,amp = blob

        #generate a slice
        myslice = self.sliceMaker(y,x,fitwidth)

        #save the upper left coordinates for later use
        ystart = myslice[0].start
        xstart = myslice[1].start

        #insert the equivalent of `:` at the beginning
        myslice.insert(0,slice(None, None, None))

        #pull the substack
        substack = self.stack[myslice]

        #fit the max projection for a good initial guess
        max_z = Gauss2D(substack.max(0))
        max_z.optimize_params_ls(**kwargs)

        #save the initial guess for later use
        guess_params = max_z.opt_params

        #check to see if initial fit was successful, if so proceed
        if np.isfinite(guess_params).all():

            def get_params(myslice):
                '''
                A helper function for the list comprehension below

                Takes a slice and fits a gaussian to it, makes sure to update
                fit window coordinates to full ROI coordinates
                '''

                #set up the fit object
                fit = Gauss2D(myslice)

                #do the fit, using the guess_parameters
                fit.optimize_params_ls(guess_params = guess_params, quiet=quiet,**kwargs)

                #get the optimized parameters as a dict
                opt = fit.opt_params_dict()

                #update coordinates
                opt['x0']+=xstart
                opt['y0']+=ystart

                #add an estimate of the noise
                opt['noise']= (myslice-fit.fit_model).std()

                #return updated coordinates
                return opt

            #prep our container
            peakfits = [get_params(myslice) for myslice in substack]

            #turn everything into a data frame for easy manipulation.
            peakfits_df = pd.DataFrame(peakfits)
            #convert sigmas to positive values
            peakfits_df[['sigma_x','sigma_y']] = abs(peakfits_df[['sigma_x','sigma_y']])
            peakfits_df.index.name = 'slice'



            return peakfits_df
        else:
            #initial fit failed, return None
            if not quiet:
                print('blob {} is unfittable'.format(blob))
            return None

    def fitPeaks(self, *args, **kwargs):
        super().fitPeaks(*args,**kwargs)
        ni = pd.MultiIndex.from_product([np.arange(self.norients),\
         np.arange(self.nphases)],names=['orientation','phase'])

        for peak in self.fits:
            peak['ni']=ni
            peak.set_index('ni',inplace=True)

        self.fits = [peak.reindex(ni) for peak in self.fits]

        return self.fits

    def calc_sim_params(self,modtype = 'nl',**kwargs):
        fits = self.fits
        periods = self.periods

        def calc_mod(data):
            '''
            A utility function to calculate modulation depth

            This is really a place holder until the linear prediction method
            can be implemented.

            1 is full modulation depth, 0 is none.
            '''
            #calculate the standard deviation
            s = np.nanstd(data)
            #calculate the mean
            m = np.nanmean(data)
            #filter data, note that the amplitude of a sinusoid is sqrt(2)*std
            #our filter band is a little bigger
            #NOTE: we could use masked arrays here.
            fdata = data.copy()
            fdata[np.abs(data-m) > 1.5*np.sqrt(2)*s] = np.nan
            #calculate the modulation depth and return it
            mod = (np.nanmax(fdata)-np.nanmin(fdata))/np.nanmax(fdata)

            return {"modulation" : mod}

        def calc_mod2(data):
            '''
            Need to change this so that it:
            - first tries to fit only the amplitude and phase
                - if that doesn't work, estimate amp and only fit phase
            - then do full fit
            '''

            #pull internal number of phases
            nphases = self.nphases

            mod = np.nan
            opt_a = np.nan
            opt_f = np.nan
            opt_p = np.nan
            opt_o = np.nan
            res = np.nan
            SNR = np.nan

            #only deal with finite data
            #NOTE: could use masked wave here.
            finite_args = np.isfinite(data)
            data_fixed = data[finite_args]

            if len(data_fixed) > 4:
                #we can't fit data with less than 4 points

                #make x-wave
                x = np.arange(nphases)[finite_args]

                #make guesses
                #amp of sine wave is sqrt(2) the standard deviation
                g_a = np.sqrt(2)*(data_fixed.std())
                #offset is mean
                g_o = data_fixed.mean()
                #frequency is such that `nphases` covers `periods`
                g_f = periods/nphases
                #guess of phase is from first data point (maybe mean of all?)
                g_p = np.arcsin((data_fixed[0]-g_o)/g_a)-2*np.pi*g_f*x[0]
                #make guess sequence
                pguess = (g_a,g_f,g_p,g_o)

                try:
                    popt,pcov = curve_fit(sine,x,data_fixed,p0=pguess)
                except RuntimeError as e:
                    pass
                    # print(e)
                    #if fit fails, put nan
                    # mod = np.nan
                    # opt_a = np.nan
                    # opt_f = np.nan
                    # opt_p = np.nan
                    # opt_o = np.nan
                except TypeError as e:
                    print(e)
                    print(data_fixed)
                    # mod = np.nan
                    # opt_a = np.nan
                    # opt_f = np.nan
                    # opt_p = np.nan
                    # opt_o = np.nan
                else:
                    opt_a,opt_f,opt_p,opt_o = popt
                    opt_a = np.abs(opt_a)
                    #if any part of the fit is negative, mark as failure
                    if opt_o - opt_a > 0:
                    #     mod = np.nan
                    # else:
                    #     #calc mod
                        mod = 2*opt_a/(opt_o+opt_a)
                        res = data_fixed-sine(x,*popt)
                        SNR = (opt_o+opt_a)/res.std()
            # else:


            return {'modulation' : mod, 'amp' : opt_a, 'freq' : opt_f,
            'phase' : opt_p, 'offset' : opt_o}

        sim_params = []

        for fit in fits:
            for i, trace in fit.dropna().groupby(level='orientation'):

                #pull amplitude values
                if modtype =='nl':
                    params = calc_mod2(trace.amp.values)
                else:
                    params = calc_mod(trace.amp.values)

                #take mean and pass to dict
                temp = trace.mean().to_dict()
                #add orientation and modulation
                temp['orientation'] = i
                #copy params over to temp for output
                for k, v in params.items():
                    temp[k] = v

                temp['SNR'] = np.median((trace.amp/trace.noise))
                #calc the SNR using the noise from the fit
                sim_params.append(temp)

        self.sim_params = pd.DataFrame(sim_params)

    def plot_sim_params(self, orientations = None, **kwargs):
        sim_params = self.sim_params
        norients = self.norients
        fig, ax = plt.subplots(1, norients, figsize=(4*norients, 4))

        for i, orient in sim_params.groupby('orientation'):
            orient = orient.dropna()
            if orientations is not None:
                name = orientations[i]
            else:
                name = i

            scatterplot(orient.modulation.values, orient.y0.values,
            orient.x0.values, ax = ax[i], fig=fig, **kwargs)
            ax[i].set_title('Orientation {}, avg mod = {:.3f}'.format(name,orient.modulation.mean()))

        fig.tight_layout()

        return fig, ax

    def calc_modmap(self):
        nphases = self.nphases
        norients = self.norients
        stack = self.stack
        periods = self.periods

        #reshape stack
        #remember that stack is phases*angles, y, x
        nphase_angle, ny, nx = stack.shape

        #check to make sure our dimensions match
        assert nphase_angle == nphases*norients

        new_stack = stack.reshape(norients, nphases, ny, nx)
        #if we wanted to follow SIMCheck completely we'd have an Anscombe
        #transform here #((2 * Math.sqrt((double)ab[a][b])) + (3.0d / 8))
        #new_stack = Anscombe(new_stack)
        #fourier transform along phases
        fft_new_stack = np.abs(fft(new_stack, axis=1))

        dc_stack = fft_new_stack[:, 0]
        #amplitudes are the 1st freq bin, because if you run 2D-SIM properly
        #you cover exactly one period, this is accounted for with the instance
        # property `periods`
        amp_stack = fft_new_stack[:, periods]
        if nphases > 3:
            #the highest frequency component is expected to be dominated by noise
            noise_stack = fft_new_stack[:, nphases//2]
            #average along angles and divide by average noise.
            self.MCNR = amp_stack.mean(0)/noise_stack.mean()

        #return the Amp-to-DC ratio. Max value should be 0.5
        #the returned stack is ordered by angle
        self.ADCR = amp_stack/dc_stack

    @property
    def ADCR(self):
        '''
        Amplitude to DC contrast ratio.

        See doc of calc_modmap for details.

        A stack of norients images.
        '''
        #User should not be able to modify this, so return copy
        return self.ADCR

def sine(x, amp, f, p, o):
    '''
    Utility function to fit nonlinearly
    '''
    return amp*np.sin(2*np.pi*f*x+p)+o
