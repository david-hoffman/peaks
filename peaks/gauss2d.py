#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gauss2d.py
"""
Class for generating and fitting 2D Gaussian peaks.

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
from dphtools.utils.lm import curve_fit
from loguru import logger

# need basic curve fitting
from scipy.optimize import OptimizeWarning

# need measure to take image moments
from skimage.measure import moments

# need to detrend data before estimating parameters
from .utils import find_real_root_near_zero

showwarning_ = warnings.showwarning


def showwarning(message, *args, **kwargs):
    """Monkey patch showwarning to funnel to loguru."""
    logger.warning(message)
    # showwarning_(message, *args, **kwargs)


warnings.showwarning = showwarning

# Eventually we'll want to abstract the useful, abstract bits of this
# class to a parent class called peak that will allow for multiple types
# of fits
# rho = cos(theta)


class Gauss2D(object):
    """A class that encapsulates experimental data that is best modeled by a 2D gaussian peak.

    It can estimate model parameters and perform a fit to the
    data. Best fit parameters are stored in a dictionary that can be accessed
    by helper functions.

    Right now the class assumes that `data` has constant spacing
    """

    def __init__(self, data, **kwargs):
        """Represent experimental equi-spaced 2D-data best represented by a Gaussian.

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
        """Return copy of internal data."""
        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._data.copy()

    @property
    def opt_params(self):
        """Optimized parameters from the fit."""
        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._popt.copy()

    @property
    def pcov(self):
        """Covariance matrix of model parameters from the fit."""
        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._pcov.copy()

    @property
    def error(self):
        """Return whether there's an error or not."""
        if self.ier in [1, 2, 3, 4]:
            return False
        else:
            return True

    @property
    def guess_params(self):
        """Guessed parameters."""
        return self._guess_params.copy()

    #############################
    # STATIC METHOD DEFINITIONS #
    #############################
    @classmethod
    def gauss2D(cls, xdata_tuple, amp, mu0, mu1, sigma0, sigma1, rho, offset):
        """Model function for a bivariate normal distribution (not normalized).

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

        Notes
        -----
        Area = 2*amp*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)
        """
        (x0, x1) = xdata_tuple

        if x0.shape != x1.shape:
            # All functions assume that data is 2D
            raise RuntimeError("Grid is mishapen")

        if not abs(rho) < 1:
            rho = np.sign(rho) * 0.9999
            logger.warning(
                "rho cannot be greater than 1 or less than -1. Here rho is {}.".format(rho)
                + "\nCoercing to {}".format(rho)
            )

        z = (
            ((x0 - mu0) / sigma0) ** 2
            - 2 * rho * (x0 - mu0) * (x1 - mu1) / (sigma0 * sigma1)
            + ((x1 - mu1) / sigma1) ** 2
        )

        g = offset + amp * np.exp(-z / (2 * (1 - rho**2)))
        return g

    @classmethod
    def gauss2D_norot(cls, xdata_tuple, amp, x0, y0, sigma_x, sigma_y, offset):
        """Model a special case of gauss2D with rho = 0."""
        # return the general form with a rho of 0
        return cls.gauss2D(xdata_tuple, amp, x0, y0, sigma_x, sigma_y, 0.0, offset)

    @classmethod
    def gauss2D_sym(cls, xdata_tuple, amp, x0, y0, sigma_x, offset):
        """Model a special case of gauss2D_norot with sigma_x = sigma_y."""
        # return the no rotation form with same sigmas
        return cls.gauss2D_norot(xdata_tuple, amp, x0, y0, sigma_x, sigma_x, offset)

    @classmethod
    def model(cls, xdata_tuple, *args):
        """Choose the correct model function to use based on the number of arguments passed to it.

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
            raise ValueError("len(args) = {}, number out of range!".format(num_args))

    @classmethod
    def gauss2D_jac(cls, params, xdata):
        """Jacobian for full model."""
        x0 = xdata[0].ravel()
        x1 = xdata[1].ravel()
        amp, mu0, mu1, sigma0, sigma1, rho, offset = params
        # calculate the main value, minus offset
        # (derivative of constant is zero)
        value = cls.gauss2D(xdata, *params).ravel() - offset

        dydamp = value / amp

        dydmu0 = (
            value
            * ((2 * (x0 - mu0)) / sigma0**2 - (2 * rho * (x1 - mu1)) / (sigma0 * sigma1))
            / (2 * (1 - rho**2))
        )

        dydmu1 = (
            value
            * ((2 * (x1 - mu1)) / sigma1**2 - (2 * rho * (x0 - mu0)) / (sigma0 * sigma1))
            / (2 * (1 - rho**2))
        )

        dydsigma0 = (
            value
            * (
                ((x0 - mu0) ** 2 / sigma0**3)
                - ((2 * rho * (x0 - mu0) * (x1 - mu1)) / (sigma0**2 * sigma1))
            )
            / (2 * (1 - rho**2))
        )

        dydsigma1 = (
            value
            * (
                ((x1 - mu1) ** 2 / sigma1**3)
                - ((2 * rho * (x0 - mu0) * (x1 - mu1)) / (sigma1**2 * sigma0))
            )
            / (2 * (1 - rho**2))
        )

        dydrho = value * (
            ((x0 - mu0) * (x1 - mu1)) / ((1 - rho**2) * sigma0 * sigma1)
            + (
                rho
                * (
                    -((x0 - mu0) ** 2 / sigma0**2)
                    + (2 * rho * (x0 - mu0) * (x1 - mu1)) / (sigma0 * sigma1)
                    - (x1 - mu1) ** 2 / sigma1**2
                )
            )
            / ((1 - rho**2) ** 2)
        )
        # now return
        return np.vstack(
            (dydamp, dydmu0, dydmu1, dydsigma0, dydsigma1, dydrho, np.ones_like(value))
        ).T

    @classmethod
    def gauss2D_norot_jac(cls, params, xdata):
        """Jacobian for no rotation model."""
        x = xdata[0].ravel()
        y = xdata[1].ravel()
        amp, x0, y0, sigma_x, sigma_y, offset = params
        value = cls.gauss2D_norot(xdata, *params).ravel() - offset
        dydamp = value / amp
        dydx0 = value * (x - x0) / sigma_x**2
        dydsigmax = value * (x - x0) ** 2 / sigma_x**3
        dydy0 = value * (y - y0) / sigma_y**2
        dydsigmay = value * (y - y0) ** 2 / sigma_y**3
        return np.vstack((dydamp, dydx0, dydy0, dydsigmax, dydsigmay, np.ones_like(value))).T
        # the below works, but speed up only for above
        # new_params = np.insert(params, 5, 0)
        # return np.delete(cls.gauss2D_jac(new_params, xdata), 5, axis=0)

    @classmethod
    def gauss2D_sym_jac(cls, params, xdata):
        """Jacobian for symmetric model."""
        x = xdata[0].ravel()
        y = xdata[1].ravel()
        amp, x0, y0, sigma_x, offset = params
        value = cls.gauss2D_sym(xdata, *params).ravel() - offset
        dydamp = value / amp
        dydx0 = value * (x - x0) / sigma_x**2
        dydsigmax = value * (x - x0) ** 2 / sigma_x**3
        dydy0 = value * (y - y0) / sigma_x**2
        return np.vstack((dydamp, dydx0, dydy0, dydsigmax, np.ones_like(value))).T
        # new_params = np.insert(params, 4, 0)
        # new_params = np.insert(new_params, 4, params[3])
        # return np.delete(cls.gauss2D_jac(new_params, xdata), (4, 5), axis=0)

    @classmethod
    def model_jac(cls, xdata_tuple, *params):
        """Choose the correct model jacobian function to use based on the number of arguments passed to it.

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
            raise RuntimeError("len(params) = {}, number out of range!".format(num_args))

    @classmethod
    def gen_model(cls, data, *args):
        """Generate a fit if needed, useful for generating residuals.

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
        """Generate the model from this instance, if the fit hasn't been performed yet an error will be raised."""
        return self.gen_model(self._data, *self._popt)

    def area(self, **kwargs):
        """Calculate the area of the model peak.

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
            return abs(
                2
                * np.pi
                * opt_params[0]
                * opt_params[3]
                * opt_params[4]
                * np.sqrt(1 - opt_params[5] ** 2)
            )
        elif num_params == 6:
            return abs(2 * np.pi * opt_params[0] * opt_params[3] * opt_params[4])
        else:
            return abs(2 * np.pi * opt_params[0] * opt_params[3] ** 2)

    def optimize_params(
        self,
        guess_params=None,
        modeltype="norot",
        quiet=False,
        bounds=None,
        checkparams=True,
        fittype="ls",
    ):
        r"""Optimize the parameters for a 2D Gaussian model using either a least squares or maximum likelihood method.

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
            guess_params = self.estimate_params()
            if modeltype.lower() == "sym":
                guess_params = np.delete(guess_params, (4, 5))
            elif modeltype.lower() == "norot":
                guess_params = np.delete(guess_params, 5)
            elif modeltype.lower() == "full":
                pass
            else:
                raise RuntimeError("modeltype is not one of: 'sym', 'norot', 'full'")

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
                ub = np.array((np.inf,) * 5 + (1, np.inf))
                bounds = (-1 * ub, ub)
            else:
                bounds = (-np.inf, np.inf)

        if fittype.lower() == "mle":
            meth = "mle"
        elif fittype.lower() == "ls":
            # default to scipy
            meth = None
        else:
            raise RuntimeError("fittype is not one of: 'ls', 'mle'")
        try:
            popt, pcov, infodict, errmsg, ier = curve_fit(
                model_ravel,
                (xx, yy),
                data.ravel(),
                p0=guess_params,
                bounds=bounds,
                full_output=True,
                jac=self.model_jac,
                method=meth,
            )
        except RuntimeError as e:
            # print(e)
            # now we need to re-parse the error message to set all the
            # flags pull the message
            self.errmsg = e.args[0].replace("Optimal parameters not found: ", "")

            # run through possibilities for failure
            errors = {
                0: "Improper",
                5: "maxfev",
                6: "ftol",
                7: "xtol",
                8: "gtol",
                "unknown": "Unknown",
            }

            # set the error flag correctly
            for k, v in errors.items():
                if v in self.errmsg:
                    self.ier = k
        except ValueError as e:
            # This except is for bounds checking gone awry
            self.errmsg = str(e)
            self.ier = -1
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
            # if popt.size > 5:
            #     popt[3:5] = abs(popt[3:5])
            # else:
            #     popt[3] = abs(popt[3])
            self._popt = popt
            self._pcov = pcov
        else:
            logger.debug("Fitting error: " + self.errmsg)

            self._popt = guess_params * np.nan
            self._pcov = np.zeros((len(guess_params), len(guess_params))) * np.nan

        if not self.error:
            # if no fitting error calc residuals and noise
            self.residuals = self.data - self.fit_model
            self.noise = self.residuals.std()
        else:
            # if there is an error set the noise to nan
            self.noise = np.nan

        return self.opt_params

    def _check_params(self, popt):
        """Check if optimized parameters are valid and set the fit flag."""
        data = self.data

        # check to see if the gaussian is bigger than its fitting window by a
        # large amount, generally the user is advised to enlarge the fitting
        # window or disregard the results of the fit.
        sigma_msg = "Sigma larger than ROI"

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
            self.errmsg = "Amplitude unphysical, amp = {:.3f}," " data range = {:.3f}"
            # cast to float to avoid memmap problems
            self.errmsg = self.errmsg.format(popt[0], float(data.max() - data.min()))
            self.ier = 11

    def estimate_params(self):
        """Estimate the parameters that best model the data using it's moments.

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
        """
        # initialize the parameter array
        params = np.zeros(7)
        # iterate at most 10 times
        for i in range(10):
            # detrend data
            data = self._data.astype(float)
            offset = data.min()
            amp = data.max() - offset

            # calculate the moments up to second order
            M = moments(data, 2)

            # calculate model parameters from the moments
            # https://en.wikipedia.org/wiki/Image_moment# Central_moments
            ybar = M[1, 0] / M[0, 0]
            yvar = M[2, 0] / M[0, 0] - ybar**2

            xbar = M[0, 1] / M[0, 0]
            xvar = M[0, 2] / M[0, 0] - xbar**2

            covar = M[1, 1] / M[0, 0] - xbar * ybar

            # place the model parameters in the return array
            params[:3] = amp, xbar, ybar
            params[3] = np.sqrt(np.abs(xvar))
            params[4] = np.sqrt(np.abs(yvar))
            params[5] = covar / np.sqrt(np.abs(xvar * yvar))
            params[6] = offset

            if abs(params[5]) < 1:
                # if the rho is valid break the loop.
                break

        # save estimate for later use
        self._guess_params = params
        # return parameters to the caller as a `copy`, we don't want them to
        # change the internal state
        return params.copy()

    @classmethod
    def _params_dict(cls, params):
        """Return a version of params in dictionary form.

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
        keys = ["amp", "x0", "y0", "sigma_x", "sigma_y", "rho", "offset"]

        num_params = len(params)

        # adjust the dictionary size
        if num_params < 7:
            keys.remove("rho")

        if num_params < 6:
            keys.remove("sigma_y")

        return {k: p for k, p in zip(keys, params)}

    def params_errors_dict(self):
        """Return a dictionary of errors."""
        keys = ["amp_e", "x0_e", "y0_e", "sigma_x_e", "sigma_y_e", "rho_e", "offset_e"]

        # pull the variances of the parameters from the covariance matrix
        # take the sqrt to get the errors
        with np.errstate(invalid="ignore"):
            params = np.sqrt(np.diag(self.pcov))

        num_params = len(params)

        # adjust the dictionary size
        if num_params < 7:
            keys.remove("rho_e")

        if num_params < 6:
            keys.remove("sigma_y_e")

        return {k: p for k, p in zip(keys, params)}

    @classmethod
    def dict_to_params(cls, d):
        """Return a version of params in dictionary form.

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
        keys = ["amp", "x0", "y0", "sigma_x", "sigma_y", "rho", "offset"]
        values = []
        for k in keys:
            try:
                values.append(d[k])
            except KeyError:
                pass

        return np.array(values)

    def opt_params_dict(self):
        """Return dictionary of optimal parameters."""
        return self._params_dict(self.opt_params)

    def all_params_dict(self):
        """Return the parameters and their estimated errors all in one dictionary.

        NOTE: the errors will have the same key plus a '_e'
        """
        params_dict = self.opt_params_dict()
        params_dict.update(self.params_errors_dict())
        return params_dict

    def guess_params_dict(self):
        """Return dictionary of guessed parameters.

        >>> import numpy as np
        >>> myg = Gauss2D(np.random.randn(10, 10))
        >>> myg._guess_params = np.array([1, 2, 3, 4, 5, 6, 7])
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


class Gauss2Dz(Gauss2D):
    """A class that encapsulates experimental data that is best modeled by a 2D gaussian peak.

    It can estimate model parameters and perform a fit to the data. Best fit parameters are stored
    in a dictionary that can be accessed by helper functions.

    Right now the class assumes that `data` has constant spacing
    """

    def __init__(self, data, poly_coefs_df):
        """Hold experimental equi-spaced 2D-data best represented by a Gaussian.

        Parameters
        ----------
        data : array_like
            An array holding the experimental data, for now data is assumed to
            have equal spacing
        poly_coefs_df : pd.DataFrame
            A data frame holding the coefficients of polynomials

        Returns
        -------
        out : object
            A Gauss2D object holding the specified data. All other internal
            variables are internalized to `None`
        """
        # Note that we are only passing a reference to the original data here
        # so DO NOT modify this field
        super().__init__(data)

        # set up polynomial functions for relating z to sigmax and y
        self.sigma_x_poly = np.poly1d(poly_coefs_df.sigma_x)
        self.sigma_y_poly = np.poly1d(poly_coefs_df.sigma_y)
        # we need their derivatives too for the jacobian
        self.sigma_x_polyd = self.sigma_x_poly.deriv()
        self.sigma_y_polyd = self.sigma_y_poly.deriv()

    @property
    def fit_model(self):
        """Return the fitted model."""
        yy, xx = np.indices(self.data.shape)
        xdata_tuple = (xx, yy)
        # return model
        return self.model(xdata_tuple, *self._popt)

    def model(self, xdata_tuple, amp, x0, y0, z0, offset):
        """Choose the correct model function to use based on the number of arguments passed to it.

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
        args = amp, x0, y0, self.sigma_x_poly(z0), self.sigma_y_poly(z0), offset
        return self.gauss2D_norot(xdata_tuple, *args)

    def model_jac(self, xdata_tuple, *params):
        """Choose the correct model jacobian function to use based on the number of arguments passed to it.

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
        x = xdata_tuple[0].ravel()
        y = xdata_tuple[1].ravel()

        amp, x0, y0, z0, offset = params
        sigma_x, sigma_y = self.sigma_x_poly(z0), self.sigma_y_poly(z0)
        sigma_xd, sigma_yd = self.sigma_x_polyd(z0), self.sigma_y_polyd(z0)

        value = self.model(xdata_tuple, *params).ravel() - offset
        dydamp = value / amp

        dydx0 = value * (x - x0) / sigma_x**2
        dydsigmax = value * (x - x0) ** 2 / sigma_x**3

        dydy0 = value * (y - y0) / sigma_y**2
        dydsigmay = value * (y - y0) ** 2 / sigma_y**3

        dydz0 = dydsigmax * sigma_xd + dydsigmay * sigma_yd

        return np.vstack((dydamp, dydx0, dydy0, dydz0, np.ones_like(value))).T
        # the below works, but speed up only for above
        # new_params = np.insert(params, 5, 0)
        # return np.delete(cls.gauss2D_jac(new_params, xdata), 5, axis=0)

    def area(self, **kwargs):
        """Return the estimated area of the fitted peak."""
        raise NotImplementedError

    def optimize_params(
        self,
        guess_params=None,
        modeltype="norot",
        quiet=False,
        bounds=None,
        checkparams=True,
        fittype="ls",
    ):
        """Find the optimal model parameters that fit the input data."""
        # Test if we've been provided guess parameters
        # Need to test if the variable is good or not.

        if guess_params is None:
            # if not we generate them
            guess_params = self.estimate_params()

        # handle the case where the user passes a dictionary of values.
        if isinstance(guess_params, dict):
            guess_params = self.dict_to_params(guess_params)

        return super().optimize_params(
            guess_params=guess_params,
            quiet=quiet,
            bounds=bounds,
            checkparams=checkparams,
            fittype=fittype,
        )

    optimize_params.__doc__ = Gauss2D.optimize_params.__doc__

    def _check_params(self, popt):
        """Check if optimized parameters are valid and set the fit flag."""
        data = self.data
        # check to see if the amplitude makes sense
        # it must be greater than 0 but it can't be too much larger than the
        # entire range of data values
        if not (0 < popt[0] < (data.max() - data.min()) * 5):
            self.errmsg = "Amplitude unphysical, amp = {:.3f}," " data range = {:.3f}"
            # cast to float to avoid memmap problems
            self.errmsg = self.errmsg.format(popt[0], np.float(data.max() - data.min()))
            self.ier = 11

    def estimate_params(self):
        """Estimate the parameters that best model the data using it's moments.

        Returns
        -------
        params : array_like
            params[0] = amp
            params[1] = x0
            params[2] = y0
            params[3] = z0
            params[4] = offset
        """
        gauss2d_params = super().estimate_params()

        amp, x0, y0, sigma_x, sigma_y, rho, offset = gauss2d_params

        # find z estimates based on sigmas
        zx = find_real_root_near_zero(self.sigma_x_poly - sigma_x)
        zy = find_real_root_near_zero(self.sigma_y_poly - sigma_y)
        possible_z = np.array((zx, zy))
        # remove nans
        possible_z = possible_z[np.isfinite(possible_z)]
        # choose the estimate closest to zero.
        if len(possible_z):
            z0 = possible_z[np.abs(possible_z).argmin()]
        else:
            z0 = 0
        # save estimate for later use
        params = self._guess_params = np.array([amp, x0, y0, z0, offset])
        # return parameters to the caller as a `copy`, we don't want them to
        # change the internal state
        return params.copy()

    def gen_model(self, *args):
        """Generate a fit if needed, useful for generating residuals.

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
        yy, xx = np.indices(self.data.shape)
        xdata_tuple = (xx, yy)
        # return model
        return self.model(xdata_tuple, *args)

    @classmethod
    def _params_dict(cls, params):
        """Return params in dictionary form.

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
        keys = ["amp", "x0", "y0", "z0", "offset"]

        return {k: p for k, p in zip(keys, params)}

    def params_errors_dict(self):
        """Return a dictionary of errors."""
        keys = ["amp_e", "x0_e", "y0_e", "z0_e", "offset_e"]

        # pull the variances of the parameters from the covariance matrix
        # take the sqrt to get the errors
        with np.errstate(invalid="ignore"):
            params = np.sqrt(np.diag(self.pcov))

        return {k: p for k, p in zip(keys, params)}

    @classmethod
    def dict_to_params(cls, d):
        """Return a version of params in dictionary form.

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
        keys = ["amp", "x0", "y0", "z0", "offset"]
        values = []
        for k in keys:
            try:
                values.append(d[k])
            except KeyError:
                pass

        return np.array(values)


if __name__ == "__main__":
    # TODO: Make data, add noise, estimate, fit. Plot all 4 + residuals
    raise NotImplementedError
