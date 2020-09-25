"""
A set of classes for analyzing data stacks that contain punctate data
"""
import itertools as itt
import multiprocessing as mp
import os

# import time
import numpy as np
import pandas as pd
import seaborn as sns
from dphplotting import clean_grid, make_grid
from dphutils import slice_maker
from matplotlib import pyplot as plt
from matplotlib.colors import ColorConverter
from scipy import ndimage as ndi
from scipy.fftpack import fft
from scipy.ndimage.filters import median_filter
from scipy.optimize import curve_fit

from .gauss2d import Gauss2D
from .peakfinder import PeakFinder
from .utils import gauss, gauss_fit, scatterplot, sine, sine2, sine_fit, sine_jac

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import rfftfreq, rfftn

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import rfftfreq, rfftn

# TODO
# Need to move all the fitting stuff into its own class and abstract as much
# functionality from gauss2d into a parent class that can be subclassed for
# each type of peak. Hopefully regardless of dimensionality.

import logging

logger = logging.getLogger(__name__)


class StackAnalyzer(object):
    """
    A parent class for more specialized analysis classes
    """

    def __init__(self, stack):
        super().__init__()
        # stack is the image stack to be analyzed
        self.stack = stack

    def findpeaks(self):
        """
        A method to find peaks, should have data passed into it, that way child
        classes can decide how to find peaks initially.
        """
        raise NotImplementedError

    def fitPeaks(self, fitwidth, nproc=0, par_func=None, quiet=True, **kwargs):
        """
        Fit all peaks found by peak finder, has the ability to split the peaks
        among multiple processors

        Parameters
        ----------
        fitwidth : int
            Sets the size of the fitting window
        nproc : int
            number of processors to use, -1 will use all available.

        Returns
        -------
        list : list of DataFrames
            A list of DataFrames with each DataFrame holding the fits of
            one peak
        """
        blobs = self.peakfinder.blobs

        # make sure we don't try to use more processors than we have
        # if user gives -1 use all cpus
        if nproc > os.cpu_count() or nproc < 0:
            nproc = os.cpu_count()

        if nproc > 1:
            # save the data type character
            dtype_char = self.stack.dtype.char
            # allocate shared memory for the array
            shared_array_base = mp.RawArray(dtype_char, self.stack.size)
            # assign the array, this opertates through a memoryview
            # so it's very fast but has no checking
            mv_array = memoryview(shared_array_base)
            # we have to cast through bytes because of Py3 peculiarities
            mv_array.cast("B").cast(dtype_char)[:] = self.stack.ravel()
            # start pool, initilize shared array on each worker.
            with mp.Pool(nproc, _init_func, (par_func, shared_array_base, self.stack.shape)) as p:
                if not quiet:
                    logger.debug("Multiprocessing engaged with {} cores".format(nproc))
                # farm out the tasks
                results = [
                    p.apply_async(par_func, args=(fitwidth, blob, None), kwds=kwargs)
                    for blob in blobs
                ]
                # collect results
                fits = [pp.get() for pp in results]
        else:
            # serial version, just list comprehension
            fits = [par_func(fitwidth, blob, self.stack, **kwargs) for blob in blobs]

        # clear nones (i.e. unsuccessful fits)
        fits = [fit for fit in fits if fit is not None]
        self.fits = fits
        return fits

    def _calc_params(self, nproc=0, par_func=None, quiet=True, **kwargs):
        """
        Super class method to calculate parameters for child stackanalyzers

        Allows for multiprocessing, which is much easier in this case because
        there is no need for shared data. That being said we still need to put
        the processing functions _outside_ the class because we don't want to
        pickle the whole class (it still contains the `stack` object).
        """
        # pull the objects internal fits
        peakfits = self.fits
        if nproc > 1:
            # make sure we don't try to use more processors than we have
            if nproc > os.cpu_count():
                nproc = os.cpu_count()
            # spin up the pool of workers
            with mp.Pool(nproc) as p:
                if not quiet:
                    logger.debug("Multiprocessing engaged with {} cores".format(nproc))
                # farm out the tasks, because we're using module level
                # functions instead of class methods we avoid pickling
                # too much data.
                results = [
                    p.apply_async(par_func, args=(peakfit,), kwds=kwargs) for peakfit in peakfits
                ]
                # collect results
                params = [pp.get() for pp in results]
        else:
            # serial version, just list comprehension
            params = [par_func(peakfit, **kwargs) for peakfit in peakfits]
        # add peak number
        for i, param in enumerate(params):
            if param is not None:
                if isinstance(param, dict):
                    param["peak_num"] = i
                else:
                    for p in param:
                        p["peak_num"] = i
        # clear nones (i.e. unsuccessful fits)
        params = [param for param in params if param is not None]
        return params

    def drift_plot(self, title=""):
        """Plot the average change in x0 and y0"""
        # set up plot
        fig, (ax0, ax1) = plt.subplots(1, 2)
        # make holders for coordinates
        x = []
        y = []
        # loop through fits
        for f in self.fits:
            # if everything is finite add the coordinates minus bias
            if np.isfinite(f.x0).all() and np.isfinite(f.y0).all():
                x.append(f.x0 - f.x0.mean())
                y.append(f.y0 - f.y0.mean())
        assert x and y, "x or y is empty"
        # plot the mean with ci 90% bands
        sns.tsplot(x, ax=ax0, ci=90, color="b")
        sns.tsplot(y, ax=ax0, ci=90, color="r")
        ax0.set_xlabel("Frame #")
        ax0.set_ylabel("Distance (pixel)")
        # flatten data
        xar = np.array(x).ravel()
        yar = np.array(y).ravel()
        # determine good histogram bin size
        nbins = "auto"
        # plot hists
        ax1.hist(
            xar,
            color=ColorConverter().to_rgba("b", 0.5),
            normed=True,
            label="$x$",
            bins=nbins,
            histtype="stepfilled",
            range=(-1, 1),
        )
        ax1.hist(
            yar,
            color=ColorConverter().to_rgba("r", 0.5),
            normed=True,
            label="$y$",
            bins=nbins,
            histtype="stepfilled",
            range=(-1, 1),
        )
        ax1.set_xlabel("Distance (pixel)")
        ax1.legend()
        fig.suptitle(title)
        fig.tight_layout()
        return fig, (ax0, ax1)


class PSFStackAnalyzer(StackAnalyzer):
    """
    A specialized version of StackAnalyzer for PSF stacks.
    """

    def __init__(self, stack, psfwidth=1.68, **kwargs):
        super().__init__(stack)
        self.psfwidth = psfwidth
        # median filter to remove spikes
        self.peakfinder = PeakFinder(median_filter(self.stack.max(0), 3), self.psfwidth, **kwargs)
        self.peakfinder.find_blobs()
        # should have a high accuracy mode that filters the data first
        # and finds the slice with the max value before finding peaks.

    def fitPeaks(self, fitwidth, nproc=1, **kwargs):
        return super().fitPeaks(fitwidth, nproc, par_func=_fitPeaks_psf, **kwargs)

    def calc_psf_params(self, nproc=0, subrange=slice(None, None, None), **kwargs):
        """Calculate the PSF paramters for all found peaks"""
        params = super()._calc_params(
            nproc=nproc, par_func=_calc_psf_param, subrange=subrange, **kwargs
        )
        self.psf_params = pd.DataFrame(params).set_index("peak_num")
        return self.psf_params

    def plot_psf_params(self, feature="z0", **kwargs):
        psf_params = self.psf_params
        fig, ax = scatterplot(
            psf_params[feature].values, psf_params.y0.values, psf_params.x0.values, **kwargs
        )
        ax.set_title(feature)
        return fig, ax

    def diagnostic_fits(self, num=9, trim=None):
        """Diagnostic, to check if sine fits are good
        check best and worst SNR"""
        # sort sim_params by SNR
        psf_params = self.psf_params.sort_values("SNR")
        # take the top and bottom
        top_half = num // 2
        bot_half = num - top_half
        to_plot = pd.concat((psf_params.iloc[:bot_half], psf_params.iloc[-top_half:]))
        # pull internal fits for later use
        fits = self.fits
        # make a grid axes
        fig, axs = make_grid(len(to_plot))
        # loop through chosen ones
        for (peak, params), ax in zip(to_plot.iterrows(), axs.ravel()):
            # pull the amplitudes and plot
            f = fits[peak]
            amp = f.amp
            ax.plot(amp, "o")
            # calculate the fit function and display
            x = np.linspace(f.index.min(), f.index.max())
            gauss_fit = gauss(x, params.amp, params.z0, params.sigma_z, params.offset)
            ax.plot(x, gauss_fit)
            if trim:
                ax.axvline(params.z0 + trim * params.sigma_z, c="r", ls="--")
                ax.axvline(params.z0 - trim * params.sigma_z, c="r", ls="--")
            # place a title with both SNRs
            ax.set_title("SNR={:.0f}, loc={}".format(params.SNR, peak))
        # make the layout tight and return
        fig.tight_layout()
        fig, axs = clean_grid(fig, axs)
        return fig, axs


class SIMStackAnalyzer(StackAnalyzer):
    """
    docstring for SIMStackAnalyser
    """

    def __init__(self, stack, norients, nphases, psfwidth=1.68, periods=1, **kwargs):
        # make sure the stack has the right shape
        my_shape = stack.shape
        assert len(my_shape) == 3, ("Stack has wrong number of dimensions," " dim = {}").format(
            my_shape
        )
        assert stack.shape[0] == norients * nphases, (
            "Number of images does" " not equal" " orients * phases"
        )

        super().__init__(stack)

        self.psfwidth = psfwidth
        self.nphases = nphases
        self.norients = norients
        self.periods = periods

        self.peakfinder = PeakFinder(median_filter(self.stack.max(0), 3), self.psfwidth, **kwargs)
        self.peakfinder.find_blobs()
        # should have a high accuracy mode that filters the data first and
        # finds the slice with the max value before finding peaks.

    def sum_peaks(self, width):
        """
        Find peaks, then sum area around them for whole stack.

        If we're going to do this _properly_ we need a way to find areas that
        _don't_ have any beads nearby inorder to calculate noise and offset.
        """
        # fit the blobs first to find valid spots
        my_peaks = self.peakfinder

        peakfits = my_peaks.fit_blobs(diameter=width)
        # now reset the blobs to the fit values
        my_peaks.blobs = peakfits[["y0", "x0", "sigma_x", "amp"]].values

        # label again
        my_labels = my_peaks.label_blobs(diameter=width)

        # find all the objects.
        my_objects = ndi.find_objects(my_labels)

        my_medians = np.median(self.data, axis=(1, 2))

        my_sums = np.array([self.data[:, obj[0], obj[1]].sum((1, 2)) for obj in my_objects])

        self.sums = my_sums - my_medians
        # reset blobs to original
        self.peakfinder.find_blobs()

    def fitPeaks(self, fitwidth, nproc=1, **kwargs):
        """Fit all the found peaks"""
        super().fitPeaks(fitwidth, nproc, par_func=_fitPeaks_sim, **kwargs)
        ni = pd.MultiIndex.from_product(
            [np.arange(self.norients), np.arange(self.nphases)], names=["orientation", "phase"]
        )

        for peak in self.fits:
            peak["ni"] = ni
            peak.set_index("ni", inplace=True)

        self.fits = [peak.reindex(ni) for peak in self.fits]

        return self.fits

    def calc_sim_params(self, nproc=0, modtype="ls", **kwargs):
        """Calculate all SIM parameters for found peaks"""
        # pass in the relavent parameters to the superclass
        if modtype == "ls":
            fit_func = calc_mod_ls
        elif modtype == "ls_3D":
            fit_func = calc_mod3D_ls
        elif modtype == "simple":
            fit_func = calc_mod
        else:
            raise ValueError("Unrecognized modulation fitting type " + modtype)
        params = super()._calc_params(
            nproc=nproc,
            par_func=_calc_sim_param,
            periods=self.periods,
            nphases=self.nphases,
            modtype=modtype,
            fit_func=fit_func,
            **kwargs
        )
        sim_params = pd.DataFrame(list(itt.chain.from_iterable(params)))
        if len(sim_params):
            self.sim_params = sim_params.set_index(["peak_num", "orientation"])
        return self.sim_params

    def plot_sim_params(self, orientations=None, **kwargs):
        """Make maps of the modulation depths"""
        sim_params = self.sim_params.reset_index()
        norients = self.norients
        fig, ax = plt.subplots(1, norients, figsize=(4 * norients, 4))

        for i, orient in sim_params.groupby("orientation"):
            orient = orient.dropna()
            if orientations is not None:
                name = orientations[i]
            else:
                name = i

            scatterplot(
                orient.modulation.values,
                orient.y0.values,
                orient.x0.values,
                ax=ax[i],
                fig=fig,
                **kwargs
            )
            ax[i].set_title(
                "Orientation {}, avg mod = {:.3f}".format(name, orient.modulation.mean())
            )

        fig.tight_layout()

        return fig, ax

    def plot_sim_hist(self, orientations=None, **kwargs):
        """
        Utility to plot a histogram of the SIM parameters
        """
        # pull sim_params
        sim_params = self.sim_params.reset_index()
        # check if they have any length
        if len(sim_params):
            try:
                axs = sim_params.hist(
                    bins="auto",
                    column="modulation",
                    by="orientation",
                    figsize=(4 * self.norients, 4),
                    layout=(1, self.norients),
                    histtype="stepfilled",
                    normed=True,
                    sharex=True,
                    sharey=True,
                )
            except ValueError as e:
                raise (e)
            else:
                # find out the figure, pandas doesn't return it automatically.
                fig = axs.ravel()[0].get_figure()
                # iterate through axes
                for i, ax in enumerate(axs.ravel()):
                    # if orientations are passed use them
                    if orientations is not None:
                        name = orientations[i]
                    else:
                        name = i
                    # we only expect values between 0 and 1
                    ax.set_xlim([0, 1])
                    # from the title get the location in DataFrame of desired data.
                    num = ax.title.get_text()
                    # check if the title is empyt, meaning group is empty
                    if num != "":
                        loc = int(num)
                        grouped_mods = sim_params.groupby("orientation").modulation
                        # calc mean
                        mymean = grouped_mods.mean().loc[loc]
                        # calc median
                        mymedian = grouped_mods.median().loc[loc]
                        # add lines
                        ax.axvline(mymean, color="r")
                        ax.axvline(mymedian, color="g")
                        # replace title
                        ax.set_title(
                            "{}, mean = {:.3f}, median = {:.3f}".format(name, mymean, mymedian)
                        )
                    else:
                        # if group is empty remove the axis
                        fig.delaxes(ax)
                fig.tight_layout()
                return fig, axs

    def diagnostic_fits(self, num=9):
        """Diagnostic, to check if sine fits are good
        check best and worst SNR"""
        # sort sim_params by SNR
        sim_params = self.sim_params.sort_values("SNR")
        # take the top and bottom
        top_half = num // 2
        bot_half = num - top_half
        to_plot = pd.concat((sim_params.iloc[:bot_half], sim_params.iloc[-top_half:]))
        # pull internal fits for later use
        fits = self.fits
        # make a grid axes
        fig, axs = make_grid(len(to_plot))
        # loop through chosen ones
        for ((peak, orient), params), ax in zip(to_plot.iterrows(), axs.ravel()):
            # pull the amplitudes and plot
            amp = fits[peak].loc[orient].amp
            ax.plot(amp, "o")
            # calculate the fit function and display
            x = np.linspace(0, len(amp) - 1)
            if "samp2" in params.keys():
                sine_fit = sine2(
                    x, params.samp, params.samp2, params.freq, params.phase, params.soffset
                )
            else:
                sine_fit = sine(x, params.samp, params.freq, params.phase, params.soffset)
            ax.plot(x, sine_fit)
            # place a title with both SNRs
            ax.set_title(
                "gSNR={:.0f}, sSNR={:.0f}, loc={}".format(
                    params.SNR, params.sin_SNR, (peak, orient)
                )
            )
        # make the layout tight and return
        fig.tight_layout()
        fig, axs = clean_grid(fig, axs)
        return fig, axs

    def calc_modmap(self):
        """WARNING, this is half baked"""
        nphases = self.nphases
        norients = self.norients
        stack = self.stack
        periods = self.periods

        # reshape stack
        # remember that stack is phases*angles, y, x
        nphase_angle, ny, nx = stack.shape

        # check to make sure our dimensions match
        assert nphase_angle == nphases * norients
        new_stack = stack.reshape(norients, nphases, ny, nx)
        # if we wanted to follow SIMCheck completely we'd have an Anscombe
        # transform here # ((2 * Math.sqrt((double)ab[a][b])) + (3.0d / 8))
        # new_stack = Anscombe(new_stack)
        # fourier transform along phases
        fft_new_stack = np.abs(fft(new_stack, axis=1))

        dc_stack = fft_new_stack[:, 0]
        # amplitudes are the 1st freq bin, because if you run 2D-SIM properly
        # you cover exactly one period, this is accounted for with the instance
        # property `periods`
        amp_stack = fft_new_stack[:, periods]
        if nphases > 3:
            # the highest frequency component is expected to be dominated
            # by noise
            noise_stack = fft_new_stack[:, nphases // 2]
            # average along angles and divide by average noise.
            self.MCNR = amp_stack.mean(0) / noise_stack.mean()

        # return the Amp-to-DC ratio. Max value should be 0.5
        # the returned stack is ordered by angle
        self.ADCR = amp_stack / dc_stack

    @property
    def ADCR(self):
        """
        Amplitude to DC contrast ratio.

        See doc of calc_modmap for details.

        A stack of norients images.
        """
        # User should not be able to modify this, so return copy
        return self.ADCR


def fitPeak(stack, slices, width, startingfit, **kwargs):
    """
    Method to fit a peak through the stack.

    The method will track the peak through the stack, assuming that moves
    are relatively small from one slice to the next

    Parameters
    ----------
    slices : iterator
        an iterator which dictates which slices to fit, should yeild
        integers only

    width : integer
        width of fitting window

    startingfit : dict
        fit coefficients

    Returns
    -------
    list : list of dicts
        A list of dictionaries containing the best fits. Easy to turn into
        a DataFrame

    """
    # set up our variable to return
    toreturn = []

    # grab the starting fit parameters
    popt_d = startingfit.copy()

    y0 = int(round(popt_d["y0"]))
    x0 = int(round(popt_d["x0"]))

    if len(popt_d) == 6 * 2:
        modeltype = "norot"
    elif len(popt_d) == 5 * 2:
        modeltype = "sym"
    elif len(popt_d) == 7 * 2:
        modeltype = "full"
    else:
        raise ValueError("Dictionary is too big {}".format(popt_d))

    for s in slices:

        # make the slice
        try:
            myslice = slice_maker((y0, x0), width)
        except RuntimeError as e:
            logger.warning("Fit window moved to edge of ROI")
            break
        else:
            # pull the starting values from it
            ystart = myslice[0].start
            xstart = myslice[1].start

            # insert the z-slice number
            myslice = (s,) + myslice

            # set up the fit and perform it using last best params
            sub_stack = stack[myslice]
            if sub_stack.size == 0:
                # the fir window has moved to the edge, break
                logger.warning("Fit window moved to edge of ROI")
                break
            fit = Gauss2D(sub_stack)

            # move our guess coefs back into the window
            popt_d["x0"] -= xstart
            popt_d["y0"] -= ystart
            # leave this in for now for easier debugging in future.
            try:
                fit.optimize_params(popt_d, **kwargs)
            except TypeError as e:
                print(repr(myslice))
                raise e

            # if there was an error performing the fit, try again without
            # a guess
            if fit.error:
                fit.optimize_params(modeltype=modeltype, **kwargs)

            # if there's not an error update center of fitting window and
            # move on to the next fit
            if not fit.error:
                popt_d = fit.all_params_dict()
                popt_d["x0"] += xstart
                popt_d["y0"] += ystart

                popt_d["slice"] = s
                # calculate the apparent noise as the standard deviation
                # of what's the residuals of the fit
                popt_d["noise"] = (sub_stack - fit.fit_model).std()
                toreturn.append(popt_d.copy())

                y0 = int(round(popt_d["y0"]))
                x0 = int(round(popt_d["x0"]))
            else:
                # if the fit fails, make sure to _not_ update positions.
                bad_fit = fit.all_params_dict()
                bad_fit["slice"] = s
                # noise of a failed fit is not really useful
                popt_d["noise"] = np.nan

                toreturn.append(bad_fit.copy())

    return toreturn


def _fitPeaks_psf(fitwidth, blob, stack, **kwargs):
    """Fitting subfucntion for PSFStackAnalyzer"""
    # check if we're being dispatched from the multiprocessing pool
    if stack is None:
        stack = _fitPeaks_psf.stack
    # unpack peak variables
    y, x, w, amp = blob
    # make the slice around the blob
    myslice = slice_maker((y, x), fitwidth)
    # find the start
    ystart = myslice[0].start
    xstart = myslice[1].start
    # insert the equivalent of `:` at the beginning
    myslice = (slice(None, None, None),) + myslice
    # make the substack
    substack = stack[myslice]
    # we could do median filtering on the substack before attempting to
    # find the max slice!
    # this could still get messed up by salt and pepper noise.
    # my_max = np.unravel_index(substack.argmax(), substack.shape)
    # use the range of each z-slice as an indication of intensity
    my_max = (substack.max((1, 2)) - substack.min((1, 2))).argmax()
    # now change my slice to be that zslice
    myslice = (my_max,) + myslice[1:]
    substack = stack[myslice]
    # prep our container
    peakfits = []
    # initial fit
    max_z = Gauss2D(substack)
    max_z.optimize_params(**kwargs)

    if np.isfinite(max_z.opt_params).all():

        # recenter the coordinates and add a slice variable
        opt_params = max_z.all_params_dict()
        opt_params["slice"] = my_max
        opt_params["x0"] += xstart
        opt_params["y0"] += ystart

        # append to our list
        peakfits.append(opt_params.copy())

        # pop the slice parameters
        opt_params.pop("slice")

        forwardrange = range(my_max + 1, stack.shape[0])
        backwardrange = reversed(range(0, my_max))

        peakfits += fitPeak(stack, forwardrange, fitwidth, opt_params.copy(), quiet=True)

        peakfits += fitPeak(stack, backwardrange, fitwidth, opt_params.copy(), quiet=True)

        # turn everything into a data frame for easy manipulation.
        peakfits_df = pd.DataFrame(peakfits)
        # convert sigmas to positive values
        try:
            peakfits_df[["sigma_x", "sigma_y"]] = abs(peakfits_df[["sigma_x", "sigma_y"]])
        except KeyError:
            peakfits_df["sigma_x"] = abs(peakfits_df["sigma_x"])

        return peakfits_df.set_index("slice").sort_index()
    else:
        logger.warning("blob {} is unfittable".format(blob))
        return None


def _fitPeaks_sim(fitwidth, blob, stack, **kwargs):
    """
    A sub function that can be dispatched to multiple cores for processing

    This function is specific to analyzing SIM data and is designed to fit
    substacks _without_ moving the fit window (i.e. it is assumed that
    drift is minimal).

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
    """
    # fix stack
    if stack is None:
        # if stack is None we know we've been decorated
        stack = _fitPeaks_sim.stack
    # pull parameters from the blob
    y, x, w, amp = blob

    # generate a slice
    myslice = slice_maker((y, x), fitwidth)

    # save the upper left coordinates for later use
    ystart = myslice[0].start
    xstart = myslice[1].start

    # insert the equivalent of `:` at the beginning
    myslice = (slice(None, None, None),) + myslice

    # pull the substack
    substack = stack[myslice]

    # fit the max projection for a good initial guess
    max_z = Gauss2D(substack.max(0))
    max_z.optimize_params(**kwargs)

    # save the initial guess for later use
    guess_params = max_z.opt_params

    # check to see if initial fit was successful, if so proceed
    if np.isfinite(guess_params).all():

        def get_params(myslice):
            """
            A helper function for the list comprehension below

            Takes a slice and fits a gaussian to it, makes sure to update
            fit window coordinates to full ROI coordinates
            """

            # set up the fit object
            fit = Gauss2D(myslice)

            # do the fit, using the guess_parameters
            fit.optimize_params(guess_params=guess_params, **kwargs)

            # get the optimized parameters as a dict
            opt = fit.all_params_dict()

            # update coordinates
            opt["x0"] += xstart
            opt["y0"] += ystart

            # add an estimate of the noise
            opt["noise"] = (myslice - fit.fit_model).std()

            # return updated coordinates
            return opt

        # prep our container
        peakfits = [get_params(myslice) for myslice in substack]

        # turn everything into a data frame for easy manipulation.
        peakfits_df = pd.DataFrame(peakfits)
        # convert sigmas to positive values
        peakfits_df[["sigma_x", "sigma_y"]] = abs(peakfits_df[["sigma_x", "sigma_y"]])
        peakfits_df.index.name = "slice"

        return peakfits_df
    else:
        # initial fit failed, return None
        return None


def _calc_psf_param(fit, subrange=slice(None, None, None), **kwargs):
    # pull values from DataFrame
    tempfit = fit.dropna().loc[subrange]
    if len(tempfit) < 4:
        logger.warning("Not enough points to fit in _calc_psf_param")
        return None
    z = tempfit.index.values
    # amp, x, y, s_x, s_y = tempfit[
    #     ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y']
    # ].values.T

    # TODO
    # need to make this robust to different fitting models.

    # do the fit to a gaussian
    popt, pcov = gauss_fit(z, tempfit.amp, **kwargs)

    # if the fit has not failed proceed
    if np.isfinite(popt).all():
        # pull fit parameters
        keys = ("amp", "z0", "sigma_z", "offset")
        famp, z0, sigma_z, offset = popt

        # interpolate other values (linear only)
        # x0 = np.interp(z0, z, x)
        # y0 = np.interp(z0, z, y)
        # sigma_x = np.interp(z0, z, s_x)
        # sigma_y = np.interp(z0, z, s_y)
        result = {k: np.interp(z0, z, tempfit[k]) for k in tempfit}

        noise = (tempfit.amp - gauss(z, *popt)).std()
        # add params
        result.update(dict(zip(keys, popt)))
        # add errors
        result.update({k + "_e": v for k, v in zip(keys, np.diag(pcov))})
        result["SNR"] = famp / noise
    else:
        result = None
    return result


def calc_mod(data, *args):
    """
    A utility function to calculate modulation depth

    This is really a place holder until the linear prediction method
    can be implemented.

    1 is full modulation depth, 0 is none.
    """
    # calculate the standard deviation
    s = np.nanstd(data)
    # calculate the mean
    m = np.nanmean(data)
    # filter data, note that the amplitude of a sinusoid is sqrt(2)*std
    # our filter band is a little bigger
    # NOTE: we could use masked arrays here.
    fdata = data.copy()
    fdata[np.abs(data - m) > 1.5 * np.sqrt(2) * s] = np.nan
    # calculate the modulation depth and return it
    mod = (np.nanmax(fdata) - np.nanmin(fdata)) / np.nanmax(fdata)

    return {"modulation": mod}


# modamp is defined as the maximum of the signal divided by the max - min
# i.e. mod = (max - min) / max when min is 0 we have perfect modulation
# depth (1) when min = max we have the worst modulation depth (0)

# we can also to a straight linear regression if we assume we know the period
# (which we should) http://stats.stackexchange.com/questions/257785/phase-modelling-while-fitting-sine-wave-to-cyclic-data


def calc_mod_ls(data, periods, nphases):
    """
    Need to change this so that it:
    - first tries to fit only the amplitude and phase
        - if that doesn't work, estimate amp and only fit phase
    - then do full fit

    Also, add a Jacobian for the curve_fit
    """
    try:
        # Fit the sine wave
        popt, pcov = sine_fit(data, periods)
    except (RuntimeError, TypeError) as e:
        print(e)
    else:
        opt_a, opt_f, opt_p, opt_o = popt
        if opt_a < 0:
            opt_a = np.abs(opt_a)
            opt_p += np.pi
        # if any part of the fit is negative, mark as failure
        mod = 2 * opt_a / (opt_o + opt_a)
        if 1 >= mod >= 0:
            res = data - sine(np.arange(len(data)), *popt)
            SNR = (opt_a) / np.nanstd(res)
            return {
                "modulation": mod,
                "samp": opt_a,
                "freq": opt_f,
                "phase": opt_p,
                "soffset": opt_o,
                "sin_SNR": SNR,
            }
    # return None


def _estimate_sine_params(data, periods, nphases):
    """utility to estimate sine params"""
    pnts = np.arange(3) * periods
    fft_data = rfftn(data)
    g_o, g_a, g_a2 = abs(fft_data[pnts]) / len(data)
    if g_a > g_a2:
        g_p = np.angle(fft_data[1]) / periods
    else:
        g_p = np.angle(fft_data[2]) / (2 * periods)
    g_f = 1 / nphases
    return g_a, g_a2, g_f, g_p, g_o


def calc_mod3D_ls(data, periods, nphases):
    """
    Need to change this so that it:
    - first tries to fit only the amplitude and phase
        - if that doesn't work, estimate amp and only fit phase
    - then do full fit

    Also, add a Jacobian for the curve_fit
    """
    # only deal with finite data
    # NOTE: could use masked wave here.
    finite_args = np.isfinite(data)
    data_fixed = data[finite_args]

    if len(data_fixed) > 4:
        # we can't fit data with less than 4 points

        # make x-wave
        x = np.arange(nphases)[finite_args]

        # make guesses
        pguess = _estimate_sine_params(np.nan_to_num(data), periods, nphases)

        try:
            # The jacobian actually slows down the fitting my guess is there
            # aren't generally enough points to make it worthwhile
            popt, pcov = curve_fit(sine2, x, data_fixed, p0=pguess)
            # popt, pcov = curve_fit(sine, x, data_fixed, p0=pguess,
            #                        Dfun=sine_jac, col_deriv=True)
        except RuntimeError as e:
            pass
        except TypeError as e:
            print(e)
            print(data_fixed)
        else:
            opt_a, opt_a2, opt_f, opt_p, opt_o = popt
            if np.sign(opt_a) != np.sign(opt_a2):
                # signs must match, waves must be in phase
                return None
            if opt_a < 0:
                opt_a = np.abs(opt_a)
                opt_a2 = np.abs(opt_a2)
                opt_p = np.abs(np.pi)
            # # if any part of the fit is negative, mark as failure
            # if (opt_a + opt_a2) * 2 > opt_o:
            # modulation for a + b * cos(x) + c * cos(2*x) is more complicated ...
            mod = (opt_a + 4 * opt_a2) ** 2 / (opt_o + opt_a + opt_a2) / (8 * opt_a2)
            res = data_fixed - sine2(x, *popt)
            SNR = (opt_a + opt_a2) / res.std()
            return {
                "modulation": mod,
                "samp": opt_a,
                "samp2": opt_a2,
                "freq": opt_f,
                "phase": opt_p,
                "soffset": opt_o,
                "sin_SNR": SNR,
            }


def _calc_sim_param(fit, *, periods, nphases, modtype, fit_func, **kwargs):
    """Function to calculate the SIM parameters for a found peak
    i.e. modulation depth"""
    results = []
    for i, trace in fit.groupby(level="orientation"):
        # pull amplitude values
        params = fit_func(trace.amp.values, periods, nphases)
        if params:
            # take mean and pass to dict
            result = trace.mean().to_dict()
            # add orientation and modulation
            result["orientation"] = i
            # copy params over to temp for output
            result.update(params)
            # calc the SNR using the noise from the fit
            result["SNR"] = np.nanmedian((trace.amp / trace.noise))
            results.append(result)
    # need to flatten the list, next function will try to make a DataFrame
    return results


def _init_func(func, stack, shape):
    """A utility function that decorates `func` to hold the
    shared stack as an accessable numpy array-like variable."""
    func.stack = np.ctypeslib.as_array(stack)
    func.stack.shape = shape
