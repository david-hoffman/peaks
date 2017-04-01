#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gauss2d.py
"""
Class for generating and fitting 2D Gaussian peaks

Supports both least squares and MLE fitting and gaussian peaks
parameterized by a single width, widths along each axis and widths
along arbitrary axes. Fitting can be done with manually specified
guesses or initial guesses will be estimated from the data.

Supports parameter passing through dicts and tuples.

Copyright (c) 2016, David Hoffman
"""

# need to be able to deal with warnings
import warnings
# numpy for numerical
import numpy as np
# need measure to take image moments
from skimage.measure import moments
# need basic curve fitting
from scipy.optimize import OptimizeWarning
# import curve_fit in a way that we can monkey patch it.
import scipy.optimize.minpack as mp
from scipy.linalg import solve_triangular
# need to detrend data before estimating parameters
from .utils import detrend, _ensure_positive

# Eventually we'll want to abstract the useful, abstract bits of this
# class to a parent class called peak that will allow for multiple types
# of fits
# rho = cos(theta)


# Functions to monkey patch in...
def _general_function_mle(params, xdata, ydata, function):
    # calculate the function
    f = function(xdata, *params)
    # calculate the MLE version of chi2
    f, ydata = _ensure_positive(f), _ensure_positive(ydata * 1.0)
    chi2 = 2 * (f - ydata - ydata * np.log(f / ydata))
    if chi2.min() < 0:
        # jury rigged to enforce positivity
        # once scipy 0.17 is released this won't be necessary.
        # don't know what the above comment means ...
        warnings.warn("Chi^2 is less than 0")
        return np.sqrt(np.nan_to_num(np.inf)) * np.ones_like(chi2)
    else:
        # return the sqrt because the np.leastsq will square and sum the result
        return np.sqrt(chi2)


def _wrap_func_mle(func, xdata, ydata, transform):
    if transform is None:
        def func_wrapped(params):
            return _general_function_mle(params, xdata, ydata, func)
    elif transform.ndim == 1:
        def func_wrapped(params):
            return transform * (_general_function_mle(params, xdata, ydata, func))
    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd)
        # by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        def func_wrapped(params):
            return solve_triangular(transform, _general_function_mle(params, xdata, ydata, func), lower=True)
    return func_wrapped


def _wrap_func_ls(func, xdata, ydata, transform):
    if transform is None:
        def func_wrapped(params):
            return func(xdata, *params) - ydata
    elif transform.ndim == 1:
        def func_wrapped(params):
            return transform * (func(xdata, *params) - ydata)
    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd)
        # by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        def func_wrapped(params):
            return solve_triangular(transform, func(xdata, *params) - ydata, lower=True)
    return func_wrapped


class Gauss2D(object):
    """
    A class that encapsulates experimental data that is best modeled by a 2D
    gaussian peak. It can estimate model parameters and perform a fit to the
    data. Best fit parameters are stored in a dictionary that can be accessed
    by helper functions.

    Right now the class assumes that `data` has constant spacing
    """

    def __init__(self, data, **kwargs):
        """
        Holds experimental equi-spaced 2D-data best represented by a Gaussian

        Parameters
        ----------
        data : array_like
            An array holding the experimental data, for now data is assumed to
            have equal spacing

        Returns
        -------
        out : object
            A Gauss2D object holding the specified data. All other internal
            variables are internalized to `None`
        """

        # Note that we are only passing a reference to the original data here
        # so DO NOT modify this field
        self._data = data
        # set all internal fields to point to NONE
        self._guess_params = None
        self._popt = None
        self._pcov = None
        self.errmsg = None
        self.ier = None
        self.noise = None
        self.residuals = None
        super().__init__(**kwargs)

    ########################
    # PROPERTY DEFINITIONS #
    ########################
    @property
    def data(self):
        """
        Internal data
        """

        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._data.copy()

    @property
    def opt_params(self):
        """
        Optimized parameters from the fit
        """

        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._popt.copy()

    @property
    def pcov(self):
        """
        Covariance matrix of model parameters from the fit
        """

        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._pcov.copy()

    @property
    def error(self):
        """Gives whether there's an error or not."""
        if self.ier in [1, 2, 3, 4]:
            return False
        else:
            return True

    @property
    def guess_params():
        """Guessed parameters"""
        return self._guess_params.copy()

    #############################
    # STATIC METHOD DEFINITIONS #
    #############################
    @classmethod
    def gauss2D(cls, xdata_tuple, amp, mu0, mu1, sigma0, sigma1, rho, offset):
        """
        A model function for a bivariate normal distribution (not normalized)

        see http://mathworld.wolfram.com/BivariateNormalDistribution.html for
        details

        Parameters
        ----------
        xdata_tuple : tuple of array_like objects
            First element is x0 and second is x1, each usually from np.meshgrid
            x0 and x1 must have the same shape
        amp : float
            Amplitude
        mu0 : float
            center x position
        mu1 : float
            center y position
        sigma0 : float
            x width
        sigma1 : float
            y width
        rho : float
            correlation between x and y (defines the angle the distributions
            major axes make with the coordinate system)
        offset : float
            offset

        Returns
        -------
        g : array_like
            A matrix of values that represent a 2D Gaussian peak. `g` will have
            the same dimensions as `x0` and `x1`

        Note: Area = 2*amp*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)
        """

        (x0, x1) = xdata_tuple

        if x0.shape != x1.shape:
            # All functions assume that data is 2D
            raise RuntimeError("Grid is mishapen")

        if not abs(rho) < 1:
            rho = np.sign(rho) * 0.9999
            warnings.warn(
                'rho cannot be greater than 1 or less than -1. Here rho is {}.'
                .format(rho) +
                '\nCoercing to {}'
                .format(rho))

        z = (((x0 - mu0) / sigma0)**2 -
             2 * rho * (x0 - mu0) * (x1 - mu1) / (sigma0 * sigma1) +
             ((x1 - mu1) / sigma1)**2)

        g = offset + amp * np.exp(-z / (2 * (1 - rho**2)))
        return g

    @classmethod
    def gauss2D_norot(cls, xdata_tuple, amp, x0, y0, sigma_x, sigma_y, offset):
        """A special case of gauss2D with rho = 0"""
        # return the general form with a rho of 0
        return cls.gauss2D(xdata_tuple, amp, x0, y0, sigma_x, sigma_y,
                           0.0, offset)

    @classmethod
    def gauss2D_sym(cls, xdata_tuple, amp, x0, y0, sigma_x, offset):
        """A special case of gauss2D_norot with sigma_x = sigma_y"""
        # return the no rotation form with same sigmas
        return cls.gauss2D_norot(xdata_tuple, amp, x0, y0,
                                 sigma_x, sigma_x, offset)

    @classmethod
    def model(cls, xdata_tuple, *args):
        """
        Chooses the correct model function to use based on the number of
        arguments passed to it

        Parameters
        ----------
        xdata_tuple : tuple of ndarrays (xx, yy)
            The independent data

        Returns
        -------
        modeldata :

        Other Parameters
        ----------------
        *args : model parameters
        """
        num_args = len(args)

        if num_args == 5:
            return cls.gauss2D_sym(xdata_tuple, *args)
        elif num_args == 6:
            return cls.gauss2D_norot(xdata_tuple, *args)
        elif num_args == 7:
            return cls.gauss2D(xdata_tuple, *args)
        else:
            raise ValueError(
                'len(args) = {}, number out of range!'.format(num_args)
            )

    @classmethod
    def gauss2D_jac(cls, params, xdata):
        """Jacobian for full model"""
        x0 = xdata[0].ravel()
        x1 = xdata[1].ravel()
        amp, mu0, mu1, sigma0, sigma1, rho, offset = params
        # calculate the main value, minus offset
        # (derivative of constant is zero)
        value = cls.gauss2D(xdata, *params).ravel() - offset

        dydamp = value / amp

        dydmu0 = value * (
                (2*(x0-mu0))/sigma0**2 - (2*rho*(x1-mu1))/(sigma0 * sigma1)
            )/(2*(1 - rho**2))

        dydmu1 = value * (
                (2*(x1-mu1))/sigma1**2 - (2*rho*(x0-mu0))/(sigma0 * sigma1)
            )/(2*(1 - rho**2))

        dydsigma0 = value * (
                ((x0-mu0)**2/sigma0**3) -
                ((2 * rho * (x0 - mu0) * (x1 - mu1))/(sigma0**2 * sigma1))
            )/(2*(1 - rho**2))

        dydsigma1 = value * (
                ((x1-mu1)**2/sigma1**3) -
                ((2 * rho * (x0 - mu0) * (x1 - mu1))/(sigma1**2 * sigma0))
            )/(2*(1 - rho**2))

        dydrho = value*(
            ((x0-mu0)*(x1-mu1))/((1 - rho**2) * sigma0 * sigma1) +
            (rho *
             (-((x0-mu0)**2/sigma0**2) +
              (2*rho*(x0-mu0)*(x1-mu1))/(sigma0*sigma1) -
              (x1-mu1)**2/sigma1**2))/((1-rho**2)**2)
            )
        # now return
        return np.vstack((dydamp, dydmu0, dydmu1, dydsigma0, dydsigma1, dydrho,
                          np.ones_like(value))).T

    @classmethod
    def gauss2D_norot_jac(cls, params, xdata):
        """Jacobian for no rotation model"""
        x = xdata[0].ravel()
        y = xdata[1].ravel()
        amp, x0, y0, sigma_x, sigma_y, offset = params
        value = cls.gauss2D_norot(xdata, *params).ravel() - offset
        dydamp = value / amp
        dydx0 = value * (x - x0) / sigma_x**2
        dydsigmax = value * (x - x0)**2 / sigma_x**3
        dydy0 = value * (y - y0) / sigma_y**2
        dydsigmay = value * (y - y0)**2 / sigma_y**3
        return np.vstack((dydamp, dydx0, dydy0, dydsigmax,
                          dydsigmay, np.ones_like(value))).T
        # the below works, but speed up only for above
        # new_params = np.insert(params, 5, 0)
        # return np.delete(cls.gauss2D_jac(new_params, xdata), 5, axis=0)

    @classmethod
    def gauss2D_sym_jac(cls, params, xdata):
        """Jacobian for symmetric model"""
        x = xdata[0].ravel()
        y = xdata[1].ravel()
        amp, x0, y0, sigma_x, offset = params
        value = cls.gauss2D_sym(xdata, *params).ravel() - offset
        dydamp = value / amp
        dydx0 = value * (x - x0) / sigma_x**2
        dydsigmax = value * (x - x0)**2 / sigma_x**3
        dydy0 = value * (y - y0) / sigma_x**2
        return np.vstack((dydamp, dydx0, dydy0, dydsigmax,
                          np.ones_like(value))).T
        # new_params = np.insert(params, 4, 0)
        # new_params = np.insert(new_params, 4, params[3])
        # return np.delete(cls.gauss2D_jac(new_params, xdata), (4, 5), axis=0)

    @classmethod
    def model_jac(cls, xdata_tuple, *params):
        """Chooses the correct model jacobian function to use based on the
        number of arguments passed to it

        Parameters
        ----------
        xdata_tuple : tuple of ndarrays (xx, yy)
            The independent data

        Returns
        -------
        modeldata :

        Other Parameters
        ----------------
        *args : model parameters
        """
        num_args = len(params)

        if num_args == 5:
            return cls.gauss2D_sym_jac(params, xdata_tuple)
        elif num_args == 6:
            return cls.gauss2D_norot_jac(params, xdata_tuple)
        elif num_args == 7:
            return cls.gauss2D_jac(params, xdata_tuple)
        else:
            raise RuntimeError(
                'len(params) = {}, number out of range!'.format(num_args)
            )

    @classmethod
    def gen_model(cls, data, *args):
        """
        A helper method to generate a fit if needed, useful for generating
        residuals

        Parameters
        ----------
        *args : tuple
            passed directly to `model`

        Returns
        -------
        out : ndarray
            Fit generated by the model.
        """
        # generate data grid
        yy, xx = np.indices(data.shape)
        xdata_tuple = (xx, yy)
        # return model
        return cls.model(xdata_tuple, *args)

    @property
    def fit_model(self):
        """
        Generate the model from this instance, if the fit hasn't been performed
        yet an error will be raised
        """
        return self.gen_model(self._data, *self._popt)

    def area(self, **kwargs):
        """
        A function for calculating the area of the model peak

        Area = 2*amp*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)

        Parameters
        ----------
        kwargs : dictionary
            key word arguments to pass to `optimize_params`, only used if
            `opt_params` has not been caculated yet.

        Returns
        -------
        Area of the peak based on fit parameters.
        """
        # this is for convenience so that the area can
        # be returned quickly, i.e. a = Gauss2D(data).area()
        if self._popt is None:
            self.optimize_params(**kwargs)
        # extract the optimal parameters
        opt_params = self.opt_params
        num_params = len(opt_params)
        # depending on the specified model the area is calculated
        if num_params == 7:
            return abs(2 * np.pi * opt_params[0] * opt_params[3] *
                       opt_params[4] * np.sqrt(1 - opt_params[5]**2))
        elif num_params == 6:
            return abs(2 * np.pi * opt_params[0] * opt_params[3] *
                       opt_params[4])
        else:
            return abs(2 * np.pi * opt_params[0] * opt_params[3]**2)

    def optimize_params(self, guess_params=None, modeltype='norot',
                        quiet=False, bounds=None, checkparams=True,
                        detrenddata=False, fittype='ls'):
        """
        A function that will optimize the parameters for a 2D Gaussian model
        using either a least squares or maximum likelihood method

        Parameters
        ----------
        guess_params : numeric sequence, or dict (optional)
            The initial guesses for the model parameters. The number of
            parameters determines the modeltype (see notes). If no
            guesses are provided they will be estimated from the data.
            The estimation is only valid for positive data 
        modeltype : {'sym', 'norot', 'full'}, default 'norot'
            Determines the model to guess parameters for
        fittype : {'ls', 'mle'}, default 'ls'
            Specifies if a least squares fit ('ls') or maximum likelihood
            estimation ('mle') should be performed
        quiet : bool
            Determines the verbosity of the output
        bounds : (-np.inf, np.inf)
            See `scipy.optimize.curve_fit` for details, if modeltype is
            'full' then the bounds for $\rho$ are automatically set to
            (-1, 1) while the rest are left as is
        checkparams : bool
            Checks the parameters for validity after the fit, maybe replaced
            in the future by more intelligent default bounding
        detrenddata : bool
            Determines if the data should be detrended before parameter
            estimation, may be removed in the future.

        Returns
        -------
        opt_params : ndarray
            The optimized parameters from the fit. If the fit wasn't
            successful a series of np.nan's will be returned.

        Notes
        -----
        This function will call scipy.optimize to optimize the parameters of
        the model function

        MLE is for poisson noise model while LS is for gaussian noise model.
        """

        # Test if we've been provided guess parameters
        # Need to test if the variable is good or not.
        if guess_params is None:
            # if not we generate them
            guess_params = self.estimate_params(detrenddata=detrenddata)
            if modeltype.lower() == 'sym':
                guess_params = np.delete(guess_params, (4, 5))
            elif modeltype.lower() == 'norot':
                guess_params = np.delete(guess_params, 5)
            elif modeltype.lower() == 'full':
                pass
            else:
                raise RuntimeError(
                    "modeltype is not one of: 'sym', 'norot', 'full'")

        # handle the case where the user passes a dictionary of values.
        if isinstance(guess_params, dict):
            guess_params = self.dict_to_params(guess_params)

        self._guess_params = guess_params

        # pull the data attribute for use
        data = self._data

        # We need to generate the x an y coordinates for the fit
        # remember that image data generally has the higher dimension first
        # as do most python objects
        yy, xx = np.indices(data.shape)

        # define our function for fitting
        def model_ravel(*args):
            return self.model(*args).ravel()

        # TODO: We also need a function to clear nan values from data and the
        # associated xx and yy points.

        # Here we fit the data but we catch any errors and instead set the
        # optimized parameters to nan.

        # full_output is an undocumented key word of `curve_fit` if set to true
        # it returns the same output as leastsq's would, if False, as it is by
        # default it returns only popt and pcov.

        # we need to set the bounds if rho is available
        if bounds is None:
            # TODO: we can make better defaults, keep sigma_x/sigma_y positive,
            # make sure amp is positive, etc...
            # set to default for all params
            if len(guess_params) == 7:
                # make sure rho is restricted
                ub = np.array((np.inf, ) * 5 + (1, np.inf))
                bounds = (-1 * ub, ub)
            else:
                bounds = (-np.inf, np.inf)
        with warnings.catch_warnings():
            # we'll catch this error later and alert the user with a printout
            warnings.simplefilter("ignore", OptimizeWarning)

            if fittype.lower() == 'mle':
                # monkey patch in mle functions
                if not (data >= 0).all():
                    raise ValueError("Data is not non-negative, please try fittype='ls' instead")
                mp._wrap_func = _wrap_func_mle
            elif fittype.lower() == 'ls':
                # use standard ls
                mp._wrap_func = _wrap_func_ls
            else:
                raise RuntimeError("fittype is not one of: 'ls', 'mle'")
            try:
                popt, pcov, infodict, errmsg, ier = curve_fit(
                    model_ravel, (xx, yy), data.ravel(), p0=guess_params,
                    bounds=bounds, full_output=True, jac=self.model_jac)
            except RuntimeError as e:
                # print(e)
                # now we need to re-parse the error message to set all the
                # flags pull the message
                self.errmsg = e.args[0].replace(
                    'Optimal parameters not found: ', ''
                )

                # run through possibilities for failure
                errors = {0: "Improper",
                          5: "maxfev",
                          6: "ftol",
                          7: "xtol",
                          8: "gtol",
                          'unknown': "Unknown"}

                # set the error flag correctly
                for k, v in errors.items():
                    if v in self.errmsg:
                        self.ier = k

            else:
                # if we save the infodict as well then we'll start using a lot
                # of memory
                # self.infodict = infodict
                self.errmsg = errmsg
                self.ier = ier

                if checkparams:
                    self._check_params(popt)

                # check to see if the covariance is bunk
                if not np.isfinite(pcov).all():
                    self.errmsg = """
                    Covariance of the parameters could not be estimated
                    """
                    self.ier = 9

        # save parameters for later use
        # if the error flag is good, proceed
        if self.ier in [1, 2, 3, 4]:
            # make sure sigmas are positive
            if popt.size > 5:
                popt[3:5] = abs(popt[3:5])
            else:
                popt[3] = abs(popt[3])
            self._popt = popt
            self._pcov = pcov
        else:
            if not quiet:
                print('Fitting error: ' + self.errmsg)

            self._popt = guess_params * np.nan
            self._pcov = np.zeros((len(guess_params),
                                   len(guess_params))) * np.nan

        if not self.error:
            # if no fitting error calc residuals and noise
            self.residuals = self.data - self.fit_model
            self.noise = self.residuals.std()
        else:
            # if there is an error set the noise to nan
            self.noise = np.nan

        return self.opt_params

    def _check_params(self, popt):
        """
        A method that checks if optimized parameters are valid
        and sets the fit flag
        """
        data = self.data

        # check to see if the gaussian is bigger than its fitting window by a
        # large amount, generally the user is advised to enlarge the fitting
        # window or disregard the results of the fit.
        sigma_msg = 'Sigma larger than ROI'

        max_s = max(data.shape)

        if len(popt) < 6:
            if abs(popt[3]) > max_s:
                self.errmsg = sigma_msg
                self.ier = 10
        else:
            if abs(popt[3]) > max_s or abs(popt[4]) > max_s:
                self.errmsg = sigma_msg
                self.ier = 10

        # check to see if the amplitude makes sense
        # it must be greater than 0 but it can't be too much larger than the
        # entire range of data values
        if not (0 < popt[0] < (data.max() - data.min()) * 5):
            self.errmsg = ("Amplitude unphysical, amp = {:.3f},"
                           " data range = {:.3f}")
            # cast to float to avoid memmap problems
            self.errmsg = self.errmsg.format(popt[0],
                                             np.float(data.max() - data.min()))
            self.ier = 11

    def estimate_params(self, detrenddata=False):
        """
        Estimate the parameters that best model the data using it's moments

        Parameters
        ----------
        detrenddata : bool
            a keyword that determines whether data should be detrended first.
            Detrending takes *much* longer than not. Probably only useful for
            large fields of view.

        Returns
        -------
        params : array_like
            params[0] = amp
            params[1] = x0
            params[2] = y0
            params[3] = sigma_x
            params[4] = sigma_y
            params[5] = rho
            params[6] = offset

        Notes
        -----
        Bias is removed from data using detrend in the util module.
        """

        # initialize the parameter array
        params = np.zeros(7)
        # iterate at most 10 times
        for i in range(10):
            # detrend data
            if detrenddata:
                # only try to remove a plane, any more should be done before
                # passing object instatiation.
                data, bg = detrend(self._data.copy(), degree=1)
                offset = bg.mean()
                amp = data.max()
            else:
                data = self._data.astype(float)
                offset = data.min()
                amp = data.max() - offset

            # calculate the moments up to second order
            M = moments(data, 2)

            # calculate model parameters from the moments
            # https://en.wikipedia.org/wiki/Image_moment# Central_moments
            xbar = M[1, 0] / M[0, 0]
            ybar = M[0, 1] / M[0, 0]
            xvar = M[2, 0] / M[0, 0] - xbar**2
            yvar = M[0, 2] / M[0, 0] - ybar**2
            covar = M[1, 1] / M[0, 0] - xbar * ybar

            # place the model parameters in the return array
            params[:3] = amp, xbar, ybar
            params[3] = np.sqrt(np.abs(xvar))
            params[4] = np.sqrt(np.abs(yvar))
            params[5] = covar / np.sqrt(np.abs(xvar * yvar))
            params[6] = offset

            if abs(params[5]) < 1 or not detrenddata:
                # if the rho is valid or we're not detrending data,
                # break the loop.
                break

        # save estimate for later use
        self._guess_params = params
        # return parameters to the caller as a `copy`, we don't want them to
        # change the internal state
        return params.copy()

    @classmethod
    def _params_dict(cls, params):
        """
        Helper function to return a version of params in dictionary form to
        make the user interface a little more friendly

        Examples
        --------
        >>> Gauss2D._params_dict((1, 2, 3, 4, 5, 6, 7)) == {
        ...     'amp': 1,
        ...     'x0': 2,
        ...     'y0': 3,
        ...     'sigma_x': 4,
        ...     'sigma_y': 5,
        ...     'rho': 6,
        ...     'offset': 7}
        True
        """

        keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset']

        num_params = len(params)

        # adjust the dictionary size
        if num_params < 7:
            keys.remove('rho')

        if num_params < 6:
            keys.remove('sigma_y')

        return {k: p for k, p in zip(keys, params)}

    def params_errors_dict(self):
        """Return a dictionary of errors"""

        keys = [
            'amp_e',
            'x0_e',
            'y0_e',
            'sigma_x_e',
            'sigma_y_e',
            'rho_e',
            'offset_e'
        ]

        # pull the variances of the parameters from the covariance matrix
        # take the sqrt to get the errors
        with np.errstate(invalid="ignore"):
            params = np.sqrt(np.diag(self.pcov))

        num_params = len(params)

        # adjust the dictionary size
        if num_params < 7:
            keys.remove('rho_e')

        if num_params < 6:
            keys.remove('sigma_y_e')

        return {k: p for k, p in zip(keys, params)}

    @classmethod
    def dict_to_params(cls, d):
        """
        Helper function to return a version of params in dictionary form
        to make the user interface a little more friendly

        >>> Gauss2D.dict_to_params({
        ...     'amp': 1,
        ...     'x0': 2,
        ...     'y0': 3,
        ...     'sigma_x': 4,
        ...     'sigma_y': 5,
        ...     'rho': 6,
        ...     'offset': 7})
        array([1, 2, 3, 4, 5, 6, 7])
        """
        keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset']
        values = []
        for k in keys:
            try:
                values.append(d[k])
            except KeyError:
                pass

        return np.array(values)

    def opt_params_dict(self):
        return self._params_dict(self.opt_params)

    def all_params_dict(self):
        """Return the parameters and there estimated errors all in one dictionary
        the errors will have the same key plus a '_e'"""
        params_dict = self.opt_params_dict()
        params_dict.update(self.params_errors_dict())
        return params_dict

    def guess_params_dict(self):
        """
        >>> import numpy as np
        >>> myg = Gauss2D(np.random.randn(10, 10))
        >>> myg.guess_params = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> myg.guess_params_dict() == {
        ...     'amp': 1,
        ...     'x0': 2,
        ...     'y0': 3,
        ...     'sigma_x': 4,
        ...     'sigma_y': 5,
        ...     'rho': 6,
        ...     'offset': 7}
        True
        """
        return self._params_dict(self.guess_params)

# we need to fix how `curve_fit` behaves with return_full
# this whole section will be removed once scipy is updated.
from scipy.linalg import svd
from scipy.optimize._lsq import least_squares
from scipy.optimize._lsq.common import make_strictly_feasible
from scipy.optimize._lsq.least_squares import prepare_bounds
def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
              check_finite=True, bounds=(-np.inf, np.inf), method=None,
              jac=None, **kwargs):
    if p0 is None:
        # determine number of parameters by inspecting the function
        from scipy._lib._util import getargspec_no_self as _getargspec
        args, varargs, varkw, defaults = _getargspec(f)
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
    else:
        p0 = np.atleast_1d(p0)
        n = p0.size

    lb, ub = prepare_bounds(bounds, n)
    if p0 is None:
        p0 = mp._initialize_feasible(lb, ub)

    bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
    if method is None:
        if bounded_problem:
            method = 'trf'
        else:
            method = 'lm'

    if method == 'lm' and bounded_problem:
        raise ValueError("Method 'lm' only works for unconstrained problems. "
                         "Use 'trf' or 'dogbox' instead.")

    # NaNs can not be handled
    if check_finite:
        ydata = np.asarray_chkfinite(ydata)
    else:
        ydata = np.asarray(ydata)

    if isinstance(xdata, (list, tuple, np.ndarray)):
        # `xdata` is passed straight to the user-defined `f`, so allow
        # non-array_like `xdata`.
        if check_finite:
            xdata = np.asarray_chkfinite(xdata)
        else:
            xdata = np.asarray(xdata)

    weights = 1.0 / np.asarray(sigma) if sigma is not None else None
    func = mp._wrap_func(f, xdata, ydata, weights)
    if callable(jac):
        jac = mp._wrap_jac(jac, xdata, weights)
    elif jac is None and method != 'lm':
        jac = '2-point'

    # Remove full_output from kwargs, otherwise we're passing it in twice.
    return_full = kwargs.pop('full_output', False)
    if method == 'lm':
        res = mp.leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
        popt, pcov, infodict, errmsg, ier = res
        cost = np.sum(infodict['fvec'] ** 2)
    else:
        res = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
                            **kwargs)

        cost = 2 * res.cost  # res.cost is half sum of squares!
        popt = res.x

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        # infodict = dict(nfev=res.nfev, fvec=res.fun, fjac=res.jac, ipvt=None,
        #                 qtf=None)
        infodict = None
        ier = res.status
        errmsg = res.message
    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("Optimal parameters not found: " + errmsg)

    warn_cov = False
    if pcov is None:
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warn_cov = True
    elif not absolute_sigma:
        if ydata.size > p0.size:
            s_sq = cost / (ydata.size - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(np.inf)
            warn_cov = True

    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov


if __name__ == '__main__':
    # TODO: Make data, add noise, estimate, fit. Plot all 4 + residuals
    raise NotImplementedError
