#Copyright David P. Hoffman

'''
Class for generating and fitting 2D Gaussian peaks
'''

#numpy for numerical
import numpy as np
#need measure to take image moments
from skimage.measure import moments
#need basic curve fitting
from scipy.optimize import curve_fit, OptimizeWarning
#need to detrend data before estimating parameters
from .utils import detrend
#need to be able to deal with warnings
import warnings
#Plotting
from matplotlib import pyplot as plt

#Eventually we'll want to abstract the useful, abstract bits of this class to a
#parent class called peak that will allow for multiple types of fits
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

        #Note that we are only passing a reference to the original data here
        #so DO NOT modify this field
        self._data = data
        self._guess_params = None
        self._popt = None
        self._pcov = None
        #self._angle = None
        super().__init__(**kwargs)

    ########################
    # PROPERTY DEFINITIONS #
    ########################

    @property
    def data(self):
        '''
        Internal data
        '''

        #This attribute should be read-only, which means that it should return
        #a copy of the data not a pointer.
        return self._data

    @property
    def opt_params(self):
        '''
        Optimized parameters from the fit
        '''

        #This attribute should be read-only, which means that it should return
        #a copy of the data not a pointer.
        return self._popt.copy()

    @property
    def pcov(self):
        '''
        Covariance matrix of model parameters from the fit
        '''

        #This attribute should be read-only, which means that it should return
        #a copy of the data not a pointer.
        return self._pcov.copy()

    @property
    def error(self):
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
            #should have error checking on parameters here
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

        Note: Area = 2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)
        '''

        (x0, x1) = xdata_tuple

        if x0.shape != x1.shape:
            #All functions assume that data is 2D
            raise ValueError

        z =((x0-mu0)/sigma0)**2 - 2*rho*(x0-mu0)*(x1-mu1)/(sigma0*sigma1) +\
         ((x1-mu1)/sigma1)**2

        g = offset + amp*np.exp( -z/(2*(1-rho**2)))
        return g

    @classmethod
    def gauss2D_norot(cls, xdata_tuple, amp, x0, y0, sigma_x, sigma_y, offset):
        '''
        A special case of gauss2D with rho = 0
        '''

        #return the general form with a rho of 0
        return cls.gauss2D(xdata_tuple, amp, x0, y0, sigma_x, sigma_y, 0.0, offset)

    @classmethod
    def gauss2D_sym(cls, xdata_tuple, amp, x0, y0, sigma_x, offset):
        '''
        A special case of gauss2D_norot with sigma_x = sigma_y
        '''

        #return the no rotation form with same sigmas
        return cls.gauss2D_norot(xdata_tuple, amp, x0, y0, sigma_x, sigma_x, offset)

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
            raise ValueError('len(args) = {}, number out of range!'.format(num_args))

        #return something in case everything is really fucked
        return -1 #should NEVER see this

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

        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])

        xx,yy = np.meshgrid(x,y)

        xdata_tuple = (xx,yy)

        return cls.model(xdata_tuple, *args)

    @property
    def fit_model(self):
        '''
        Generate the model from this instance, if the fit hasn't been performed
        yet an error will be raised
        '''
        return self.gen_model(self._data,*self._popt)

    def area(self,**kwargs):
        '''
        A function for calculating the area of the model peak

        Parameters
        ----------
        kwargs : dictionary
            key word arguments to pass to `optimize_params_ls`, only used if
            `opt_params` has not been caculated yet.

        Returns
        -------
        Area of the peak based on fit parameters.
        '''

        if self._popt is None:
            self.optimize_params_ls(**kwargs)

        opt_params = self.opt_params
        num_params = len(opt_params)

        if num_params == 7:
            return abs(2*np.pi*opt_params[0]*opt_params[3]*opt_params[4]*\
            np.sqrt(1-opt_params[5]**2))
        elif num_params == 6:
            return abs(2*np.pi*opt_params[0]*opt_params[3]*opt_params[4])
        else:
            return abs(2*np.pi*opt_params[0]*opt_params[3]**2)

    def optimize_params_ls(self, guess_params = None, modeltype = 'norot',\
                            quiet = False, check_params = True, detrenddata = False):
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

        #Test if we've been provided guess parameters
        #Need to test if the variable is good or not.
        if guess_params is None:
            #if not we generate them
            guess_params = self.estimate_params(detrenddata = detrenddata)
            if modeltype.lower() == 'sym':
                guess_params = np.delete(guess_params,(4,5))
            elif modeltype.lower() == 'norot':
                guess_params = np.delete(guess_params,5)

        #handle the case where the user passes a dictionary of values.
        try:
            guess_params = self.dict_to_params(guess_params)
        except IndexError as e:
            pass

        self._guess_params = guess_params

        #pull the data attribute for use
        data = self._data

        #We need to generate the x an y coordinates for the fit

        #remember that image data generally has the higher dimension first
        #as do most python objects
        y = np.arange(data.shape[0])
        x = np.arange(data.shape[1])
        xx, yy = np.meshgrid(x,y)

        #define our function for fitting
        def model_ravel(*args) : return self.model(*args).ravel()

        #Here we fit the data but we catch any errors and instead set the
        #optimized parameters to nan.

        #full_output is an undocumented key word of `curve_fit` if set to true
        #it returns the same output as leastsq's would, if False, as it is by
        #default it returns only popt and pcov.

        #we can use the full output to determine wether the fit was successful
        #or not. This will also allow for easier integration once MLE fitting is
        #implemented
        with warnings.catch_warnings():
            #we'll catch this error later and alert the user with a printout
            warnings.simplefilter("ignore", OptimizeWarning)

            try:
                popt, pcov, infodict, errmsg, ier= curve_fit(model_ravel, (xx, yy),\
                                data.ravel(), p0=guess_params, full_output=True)
            except RuntimeError as e:
                #print(e)
                #now we need to re-parse the error message to set all the flags
                #pull the message
                self.errmsg = e.args[0].replace('Optimal parameters not found: ','')

                #run through possibilities for failure
                errors = {0: "Improper",
                  5: "maxfev",
                  6: "ftol",
                  7: "xtol",
                  8: "gtol",
                  'unknown': "Unknown"}

                #set the error flag correctly
                for k, v in errors.items():
                    if v in self.errmsg:
                        self.ier = k

            else:
                #if we save the infodict as well then we'll start using a lot of
                #memory
                #self.infodict = infodict
                self.errmsg = errmsg
                self.ier = ier

                if check_params:
                    self._check_params(popt)

                #check to see if the covariance is bunk
                if not np.isfinite(pcov).all():
                    self.errmsg = 'Covariance of the parameters could not be estimated'
                    self.ier = 9

        #save parameters for later use
        #if the error flag is good, proceed
        if self.ier in [1, 2, 3, 4]:
            self._popt = popt
            self._pcov = pcov
        else:
            if not quiet:
                print('Fitting error: ' + self.errmsg)

            self._popt = guess_params*np.nan
            self._pcov = np.zeros((len(guess_params),len(guess_params)))*np.nan

        #return copy to user
        return self.opt_params

    def _check_params(self, popt):
        '''
        A method that checks if optimized parameters are valid sets the fit flag
        '''
        data = self.data

        #check to see if the gaussian is bigger than its fitting window by a
        #large amount, generally the user is advised to enlarge the fitting
        #window or disregard the results of the fit.
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

        #check to see if the amplitude makes sense
        #it must be greater than 0 but it can't be too much larger than the
        #entire range of data values
        if not (0 < popt[0] < (data.max() - data.min())*5):
             self.errmsg = "Amplitude unphysical"
             self.ier = 11

    def estimate_params(self,detrenddata = False):
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

        #initialize the parameter array
        params = np.zeros(7)

        #detrend data
        if detrenddata:
            #only try to remove a plane, any more should be done before passing
            #object instatiation.
            data, bg = detrend(self._data.copy(), degree = 1)
            offset = bg.mean()
            amp = data.max()
        else:
            data = self._data.astype(float)
            offset = data.min()
            amp = data.max()-offset


        #calculate the moments up to second order
        M = moments(data, 2)

        #calculate model parameters from the moments
        #https://en.wikipedia.org/wiki/Image_moment#Central_moments
        xbar = M[1,0]/M[0,0]
        ybar = M[0,1]/M[0,0]
        xvar = M[2,0]/M[0,0]-xbar**2
        yvar = M[0,2]/M[0,0]-ybar**2
        covar = M[1,1]/M[0,0]-xbar*ybar

        #place the model parameters in the return array
        params[:3] = amp, xbar, ybar
        params[3] = np.sqrt(np.abs(xvar))
        params[4] = np.sqrt(np.abs(yvar))
        params[5] = covar/np.sqrt(np.abs(xvar*yvar))
        params[6] = offset

        #save estimate for later use
        self._guess_params = params

        #return parameters to the caller as a `copy`, we don't want them to
        #change the internal state
        return params.copy()

    @classmethod
    def _params_dict(cls, params):
        '''
        Helper function to return a version of params in dictionary form to make
        the user interface a little more friendly
        '''

        keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset']

        num_params = len(params)

        #adjust the dictionary size
        if num_params < 7:
            keys.remove('rho')

        if num_params < 6:
            keys.remove('sigma_y')

        return {k : p for k,p in zip(keys,params)}

    @classmethod
    def dict_to_params(cls, d):
        '''
        Helper function to return a version of params in dictionary form to make
        the user interface a little more friendly
        '''

        keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'rho', 'offset']

        values = []
        for k in keys:
            try:
                values.append(d[k])
            except KeyError as e:
                pass

        return np.array(values)

    def opt_params_dict(self):
        return self._params_dict(self.opt_params)

    def guess_params_dict(self):
        return _params_dict(self.guess_params)

    def optimize_params_mle(self):
        print('This function has not been implemented yet, passing to\
                optimize_params_ls.')
        return optimize_params_ls(self)

    def plot(self):
        fig, ax = plt.subplots(1,1,squeeze = True)
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

if __name__ == '__main__':
    pass
    # def model_function_test():
    #     # Create x and y indices
    #     x = np.arange(64)
    #     y = np.arange(128)
    #     x, y = np.meshgrid(x, y)
    #
    #     #create data
    #     testdata = Gauss2D().gauss2D((x, y), 3, 32, 32, 5, 10, 10)
    #
    #     # add some noise to the data and instantiate object with noisy data
    #     my_gauss = Gauss2D(testdata + 0.2*np.random.randn(*testdata.shape))
    #     initial_guess = zeros(6)
    #
    #     initial_guess = my_gauss.estimate_params()
    #
    #     print(initial_guess)
    #
    #     initial_guess2d = gaussian2D((x, y), *initial_guess)
    #
    #     fig, ax = plt.subplots(1, 1)
    #     ax.hold(True)
    #     ax.matshow(testdata_noisy, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()))
    #     ax.contour(x, y, initial_guess2d, 8, colors='r')
    #
    #     popt, pcov = curve_fit(gaussian2D_fit, (x, y), testdata_noisy.ravel(), p0=initial_guess)
    #
    #     #And plot the results:
    #
    #     testdata_fitted = gaussian2D((x, y), *popt)
    #
    #     fig, ax = plt.subplots(1, 1)
    #     ax.hold(True)
    #     ax.matshow(testdata_noisy, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()))
    #     ax.contour(x, y, testdata_fitted, 8, colors='r')
