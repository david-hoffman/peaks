# Copyright David P. Hoffman

'''
Class for generating and fitting 2D Gaussian peaks
'''

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

# need to detrend data before estimating parameters
from .utils import detrend
# Plotting
from matplotlib import pyplot as plt

# Eventually we'll want to abstract the useful, abstract bits of this
# class to a parent class called peak that will allow for multiple types
# of fits
# rho = cos(theta)


# Functions to monkey patch in...
def _general_function_mle(params, xdata, ydata, function):
    # calculate the function
    f = function(xdata, *params)
    # calculate the MLE version of chi2
    chi2 = 2*(f - ydata - ydata * np.log(f/ydata))
    if chi2.min() < 0:
        # jury rigged to enforce positivity
        # once scipy 0.17 is released this won't be necessary.
        return np.nan_to_num(np.inf) * np.ones_like(chi2)
    else:
        # return the sqrt because the np.leastsq will square and sum the result
        return np.sqrt(chi2)


def _weighted_general_function_mle(params, xdata, ydata, function, weights):
    return weights * _general_function_mle(params, xdata, ydata, function)


def _general_function_ls(params, xdata, ydata, function):
    return function(xdata, *params) - ydata


def _weighted_general_function_ls(params, xdata, ydata, function, weights):
    return weights * _general_function_ls(params, xdata, ydata, function)


class Gauss2D(object):
    """
    A class that encapsulates experimental data that is best modeled by a 2D
    gaussian peak. It can estimate model parameters and perform a fit to the
    data. Best fit parameters are stored in a dictionary that can be accessed
    by helper functions.

    Right now the class assumes that `data` has constant spacing
    """

    def __init__(self, data, **kwargs):
        '''
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
        '''

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
        '''
        Internal data
        '''

        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._data.copy()

    @property
    def opt_params(self):
        '''
        Optimized parameters from the fit
        '''

        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._popt.copy()

    @property
    def pcov(self):
        '''
        Covariance matrix of model parameters from the fit
        '''

        # This attribute should be read-only, which means that it should return
        # a copy of the data not a pointer.
        return self._pcov.copy()

    @property
    def error(self):
        '''
        Gives whether there's an error or not.
        '''
        if self.ier in [1, 2, 3, 4]:
            return False
        else:
            return True

    def guess_params():
        '''
        Guess parameters for fitting.
        '''

        def fget(self):
            return self._guess_params

        def fset(self, value):
            # should have error checking on parameters here
            self._guess_params = value

        def fdel(self):
            self._guess_params = None
        return locals()
    guess_params = property(**guess_params())

    #############################
    # STATIC METHOD DEFINITIONS #
    #############################
    @classmethod
    def gauss2D(cls, xdata_tuple, amp, mu0, mu1, sigma0, sigma1, rho, offset):
        '''
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
        '''

        (x0, x1) = xdata_tuple

        if x0.shape != x1.shape:
            # All functions assume that data is 2D
            raise ValueError

        if not abs(rho) < 1:
            warnings.warn(
                'rho cannot be greater than 1 or less than -1. Here rho is {}.'
                .format(rho) +
                '\nCoercing to {}'
                .format(np.sign(rho)*0.99))

            rho = np.sign(rho)*0.99

        z = (((x0 - mu0)/sigma0)**2 -
             2*rho*(x0 - mu0)*(x1 - mu1)/(sigma0 * sigma1) +
             ((x1 - mu1)/sigma1)**2)

        g = offset + amp*np.exp(-z/(2*(1 - rho**2)))
        return g

    @classmethod
    def gauss2D_norot(cls, xdata_tuple, amp, x0, y0, sigma_x, sigma_y, offset):
        '''
        A special case of gauss2D with rho = 0
        '''

        # return the general form with a rho of 0
        return cls.gauss2D(xdata_tuple, amp, x0, y0, sigma_x, sigma_y,
                           0.0, offset)

    @classmethod
    def gauss2D_sym(cls, xdata_tuple, amp, x0, y0, sigma_x, offset):
        '''
        A special case of gauss2D_norot with sigma_x = sigma_y
        '''

        # return the no rotation form with same sigmas
        return cls.gauss2D_norot(xdata_tuple, amp, x0, y0,
                                 sigma_x, sigma_x, offset)

    @classmethod
    def model(cls, xdata_tuple, *args):
        '''
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
        '''
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
        '''
        Jacobian for full model
        '''
        x0 = xdata[0].ravel()
        x1 = xdata[1].ravel()
        amp, mu0, mu1, sigma0, sigma1, rho, offset = params
        # calculate the main value, minus offset
        # (derivative of constant is zero)
        value = cls.gauss2D(xdata, *params).ravel()-offset

        dydamp = value/amp

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
                          np.ones_like(value)))

    @classmethod
    def gauss2D_norot_jac(cls, params, xdata):
        '''
        Jacobian for no rotation model
        '''
        x = xdata[0].ravel()
        y = xdata[1].ravel()
        amp, x0, y0, sigma_x, sigma_y, offset = params
        value = cls.gauss2D_norot(xdata, *params).ravel()-offset
        dydamp = value/amp
        dydx0 = value*(x-x0)/sigma_x**2
        dydsigmax = value*(x-x0)**2/sigma_x**3
        dydy0 = value*(y-y0)/sigma_y**2
        dydsigmay = value*(y-y0)**2/sigma_y**3
        return np.vstack((dydamp, dydx0, dydy0, dydsigmax,
                          dydsigmay, np.ones_like(value)))
        # the below works, but speed up only for above
        # new_params = np.insert(params, 5, 0)
        # return np.delete(cls.gauss2D_jac(new_params, xdata), 5, axis=0)

    @classmethod
    def gauss2D_sym_jac(cls, params, xdata):
        '''
        Jacobian for symmetric model
        '''
        x = xdata[0].ravel()
        y = xdata[1].ravel()
        amp, x0, y0, sigma_x, offset = params
        value = cls.gauss2D_sym(xdata, *params).ravel()-offset
        dydamp = value/amp
        dydx0 = value*(x-x0)/sigma_x**2
        dydsigmax = value*(x-x0)**2/sigma_x**3
        dydy0 = value*(y-y0)/sigma_x**2
        return np.vstack((dydamp, dydx0, dydy0, dydsigmax,
                          np.ones_like(value)))
        # new_params = np.insert(params, 4, 0)
        # new_params = np.insert(new_params, 4, params[3])
        # return np.delete(cls.gauss2D_jac(new_params, xdata), (4, 5), axis=0)

    @classmethod
    def model_jac(cls, params, xdata_tuple, ydata, func):
        '''
        Chooses the correct model jacobian function to use based on the number
        of arguments passed to it

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
        '''
        num_args = len(params)

        if num_args == 5:
            return cls.gauss2D_sym_jac(params, xdata_tuple)
        elif num_args == 6:
            return cls.gauss2D_norot_jac(params, xdata_tuple)
        elif num_args == 7:
            return cls.gauss2D_jac(params, xdata_tuple)
        else:
            raise ValueError(
                'len(params) = {}, number out of range!'.format(num_args)
                )

    @classmethod
    def gen_model(cls, data, *args):
        '''
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
        '''

        yy, xx = np.indices(data.shape)

        xdata_tuple = (xx, yy)

        return cls.model(xdata_tuple, *args)

    @property
    def fit_model(self):
        '''
        Generate the model from this instance, if the fit hasn't been performed
        yet an error will be raised
        '''
        return self.gen_model(self._data, *self._popt)

    def area(self, **kwargs):
        '''
        A function for calculating the area of the model peak

        Parameters
        ----------
        kwargs : dictionary
            key word arguments to pass to `optimize_params`, only used if
            `opt_params` has not been caculated yet.

        Returns
        -------
        Area of the peak based on fit parameters.
        '''

        if self._popt is None:
            self.optimize_params(**kwargs)

        opt_params = self.opt_params
        num_params = len(opt_params)

        if num_params == 7:
            return abs(2 * np.pi * opt_params[0] * opt_params[3] *
                       opt_params[4] * np.sqrt(1-opt_params[5]**2))
        elif num_params == 6:
            return abs(2*np.pi*opt_params[0]*opt_params[3]*opt_params[4])
        else:
            return abs(2*np.pi*opt_params[0]*opt_params[3]**2)

    def optimize_params(self, guess_params=None, modeltype='norot',
                        quiet=False, check_params=True, detrenddata=False,
                        fit_type='ls'):
        '''
        A function that will optimize the parameters for a 2D Gaussian model
        using a least squares method

        Parameters
        ----------
        self : array_like

        Returns
        -------

        Notes
        -----
        This function will call scipy.optimize to optimize the parameters of
        the model function
        '''

        # Test if we've been provided guess parameters
        # Need to test if the variable is good or not.
        if guess_params is None:
            # if not we generate them
            guess_params = self.estimate_params(detrenddata=detrenddata)
            if modeltype.lower() == 'sym':
                guess_params = np.delete(guess_params, (4, 5))
            elif modeltype.lower() == 'norot':
                guess_params = np.delete(guess_params, 5)

        # handle the case where the user passes a dictionary of values.
        try:
            guess_params = self.dict_to_params(guess_params)
        except IndexError as e:
            pass

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

        # We also need a function to clear nan values from data and the
        # associated xx and yy points.

        # Here we fit the data but we catch any errors and instead set the
        # optimized parameters to nan.

        # full_output is an undocumented key word of `curve_fit` if set to true
        # it returns the same output as leastsq's would, if False, as it is by
        # default it returns only popt and pcov.

        # we can use the full output to determine wether the fit was successful
        # or not. This will also allow for easier integration once MLE fitting
        # is implemented
        with warnings.catch_warnings():
            # we'll catch this error later and alert the user with a printout
            warnings.simplefilter("ignore", OptimizeWarning)

            if fit_type.lower() == 'mle':
                # monkey patch in mle functions
                mp._general_function = _general_function_mle
                mp._weighted_general_function = _weighted_general_function_mle
                # upper_bound = np.ones_like(guess_params)*np.inf
                # lower_bound = np.ones_like(guess_params)*(-np.inf)
                # lower_bound[0] = 0
                # lower_bound[-1] = 0
                # bounds = (lower_bound, upper_bound)
            else:
                # use standard ls
                mp._general_function = _general_function_ls
                mp._weighted_general_function = _weighted_general_function_ls
                # bounds = (-np.inf, np.inf)

            try:
                # need to add bounds here when scipy 0.17 is released
                popt, pcov, infodict, errmsg, ier = mp.curve_fit(
                    model_ravel, (xx, yy), data.ravel(), p0=guess_params,
                    full_output=True, Dfun=self.model_jac, col_deriv=True)
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

                if check_params:
                    self._check_params(popt)

                # check to see if the covariance is bunk
                if not np.isfinite(pcov).all():
                    self.errmsg = '''
                    Covariance of the parameters could not be estimated
                    '''
                    self.ier = 9

        # save parameters for later use
        # if the error flag is good, proceed
        if self.ier in [1, 2, 3, 4]:
            self._popt = popt
            self._pcov = pcov
        else:
            if not quiet:
                print('Fitting error: ' + self.errmsg)

            self._popt = guess_params*np.nan
            self._pcov = np.zeros((len(guess_params),
                                   len(guess_params)))*np.nan

        if not self.error:
            # if no fitting error calc residuals and noise
            self.residuals = self.data-self.fit_model
            self.noise = self.residuals.std()
        else:
            # if there is an error set the noise to nan
            self.noise = np.nan

        return self.opt_params

    def _check_params(self, popt):
        '''
        A method that checks if optimized parameters are valid
        and sets the fit flag
        '''
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
        if not (0 < popt[0] < (data.max() - data.min())*5):
            self.errmsg = "Amplitude unphysical"
            self.ier = 11

    def estimate_params(self, detrenddata=False):
        '''
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
        '''

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
                amp = data.max()-offset

            # calculate the moments up to second order
            M = moments(data, 2)

            # calculate model parameters from the moments
            # https://en.wikipedia.org/wiki/Image_moment# Central_moments
            xbar = M[1, 0]/M[0, 0]
            ybar = M[0, 1]/M[0, 0]
            xvar = M[2, 0]/M[0, 0]-xbar**2
            yvar = M[0, 2]/M[0, 0]-ybar**2
            covar = M[1, 1]/M[0, 0]-xbar*ybar

            # place the model parameters in the return array
            params[:3] = amp, xbar, ybar
            params[3] = np.sqrt(np.abs(xvar))
            params[4] = np.sqrt(np.abs(yvar))
            params[5] = covar/np.sqrt(np.abs(xvar*yvar))
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
        '''
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
        '''

        keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset']

        num_params = len(params)

        # adjust the dictionary size
        if num_params < 7:
            keys.remove('rho')

        if num_params < 6:
            keys.remove('sigma_y')

        return {k: p for k, p in zip(keys, params)}

    @classmethod
    def dict_to_params(cls, d):
        '''
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
        '''

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

    def guess_params_dict(self):
        '''
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
        '''
        return self._params_dict(self.guess_params)

    def plot(self):
        fig, ax = plt.subplots(1, 1, squeeze=True)
        ax.matshow(self._data)
        return fig, ax

    def _subplot(self, params):
        fig, ax = self.plot()
        guess = self.gen_model(self.data, params)
        ax.contour(guess, color='r')

    def plot_estimated(self):
        self._subplot(self.guess_params)

    def plot_optimized(self):
        self._subplot(self.opt_params)
