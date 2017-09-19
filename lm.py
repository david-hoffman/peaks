#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lm.py
"""
A python implementation of Levenberg–Marquardt.

exposes a drop in replacement for scipy.curve_fit and
allows the user to fit their function by maximizing the
maximum likelihood for poisson deviates rather than for
gaussian deviates

### References
1. Methods for Non-Linear Least Squares Problems (2nd ed.) http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3215 (accessed Aug 18, 2017).
1. [Laurence, T. A.; Chromy, B. A. Efficient Maximum Likelihood Estimator Fitting of Histograms. Nat Meth 2010, 7 (5), 338–339.](http://www.nature.com/nmeth/journal/v7/n5/full/nmeth0510-338.html)
1. Numerical Recipes in C: The Art of Scientific Computing, 2nd ed.; Press, W. H., Ed.; Cambridge University Press: Cambridge ; New York, 1992.
1. https://www.osti.gov/scitech/servlets/purl/7256021/
Copyright (c) 2017, David Hoffman
"""

import numpy as np
from numpy import linalg as la
import scipy.optimize


def _chi2_ls(f):
    """Sum of the squares of the residuals

    Assumes that f returns residuals.

    Minimizing this will maximize the likelihood for a
    data model with gaussian deviates."""
    return 0.5 * (f**2).sum(0)


def _update_ls(x0, f, Dfun):
    """Hessian and gradient calculations for gaussian deviates"""
    # calculate the jacobian
    # j shape (ndata, nparams)
    j = Dfun(x0)
    # calculate the linear term of Hessian
    # a shape (nparams, nparams)
    a = j.T @ j
    # calculate the gradient
    # g shape (nparams,)
    g = j.T @ f
    return j, a, g


def _chi2_mle(f):
    """The equivalent "chi2" for poisson deviates

    Minimizing this will maximize the likelihood for a data
    model with gaussian deviates."""
    f, y = f
    if f.min() <= 0:
        # this is not allowed so make chi2
        # large to avoid
        return np.inf
    part1 = (f - y).sum(0)
    # don't include points where the data is less
    # than zero as this isn't allowed.
    with np.errstate(divide="ignore", invalid="ignore"):
        part2 = - (y * np.log(f / y))[y > 0].sum(0)
    return part1 + part2


def _update_mle(x0, f, Dfun):
    """Hessian and gradient calculations for poisson deviates"""
    # calculate the jacobian
    # j shape (ndata, nparams)
    f, y = f
    y_f = y / f
    j = Dfun(x0)
    # calculate the linear term of Hessian
    # a shape (nparams, nparams)
    a = ((j.T * (y_f / f)) @ j)
    # calculate the gradient
    # g shape (nparams,)
    g = j.T @ (1 - y_f)
    return j, a, g


def _wrap_func_mle(func, xdata, ydata, transform):
    """Returns f and xdata"""
    # add non-negativity constraint to data
    ydata_nn = np.maximum(ydata, 0)
    if transform is None:
        def func_wrapped(params):
            # return function and data
            return func(xdata, *params), ydata_nn
    elif transform.ndim == 1:
        raise NotImplementedError
    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd)
        # by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        raise NotImplementedError
    return func_wrapped


def _wrap_jac_mle(jac, xdata, transform):
    if transform is None:
        def jac_wrapped(params):
            return jac(xdata, *params)
    elif transform.ndim == 1:
        raise NotImplementedError
    else:
        raise NotImplementedError
    return jac_wrapped


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


def _wrap_jac_ls(jac, xdata, transform):
    if transform is None:
        def jac_wrapped(params):
            return jac(xdata, *params)
    elif transform.ndim == 1:
        def jac_wrapped(params):
            return transform[:, np.newaxis] * np.asarray(jac(xdata, *params))
    else:
        def jac_wrapped(params):
            return solve_triangular(transform, np.asarray(jac(xdata, *params)), lower=True)
    return jac_wrapped


def lm(func, x0, args=(), Dfun=None, full_output=False,
            col_deriv=True, ftol=1.49012e-8, xtol=1.49012e-8,
            gtol=0.0, maxfev=None, epsfcn=None, factor=100, diag=None, method="ls"):
    """A more thorough implementation of levenburg-marquet
    for gaussian Noise
    ::
        x = arg min(sum(func(y)**2,axis=0))
                 y
    Parameters
    ----------
    func : callable
        should take at least one (possibly length N vector) argument and
        returns M floating point numbers. It must not return NaNs or
        fitting might fail.
    x0 : ndarray
        The starting estimate for the minimization.
    args : tuple, optional
        Any extra arguments to func are placed in this tuple.
    Dfun : callable, optional
        A function or method to compute the Jacobian of func with derivatives
        across the rows. If this is None, the Jacobian will be estimated.
    full_output : bool, optional
        non-zero to return all optional outputs.
    col_deriv : bool, optional
        non-zero to specify that the Jacobian function computes derivatives
        down the columns (faster, because there is no transpose operation).
    ftol : float, optional
        Relative error desired in the sum of squares.
    xtol : float, optional
        Relative error desired in the approximate solution.
    gtol : float, optional
        Orthogonality desired between the function vector and the columns of
        the Jacobian.
    maxfev : int, optional
        The maximum number of calls to the function. If `Dfun` is provided
        then the default `maxfev` is 100*(N+1) where N is the number of elements
        in x0, otherwise the default `maxfev` is 200*(N+1).
    epsfcn : float, optional
        A variable used in determining a suitable step length for the forward-
        difference approximation of the Jacobian (for Dfun=None).
        Normally the actual step length will be sqrt(epsfcn)*x
        If epsfcn is less than the machine precision, it is assumed that the
        relative errors are of the order of the machine precision.
    factor : float, optional
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
    diag : sequence, optional
        N positive entries that serve as a scale factors for the variables.
    method : "ls" or "mle"
        What type of estimator to use. Maximum likelihood ("mle") assumes that the noise
        in the measurement is poisson distributed while least squares ("ls") assumes
        normally distributed noise.
    """
    info = 0
    x0 = np.asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    # shape, dtype = _check_func('leastsq', 'func', func, x0, args, n)
    # m = shape[0]
    # if n > m:
    #     raise TypeError('Improper input: N=%s must not exceed M=%s' % (n, m))
    if Dfun is None:
        raise NotImplementedError
        if epsfcn is None:
            epsfcn = np.finfo(dtype).eps
    else:
        if col_deriv:
            pass
            # _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (n, m))
        else:
            raise NotImplementedError("Column derivatives required")
        if maxfev is None:
            maxfev = 100 * (n + 1)

    # this is stolen from scipy.leastsq so it isn't fully implemented
    errors = {0: ["Improper input parameters.", TypeError],
              1: ["Both actual and predicted relative reductions "
                  "in the sum of squares\n  are at most %f" % ftol, None],
              2: ["The relative error between two consecutive "
                  "iterates is at most %f" % xtol, None],
              3: ["Both actual and predicted relative reductions in "
                  "the sum of squares\n  are at most %f and the "
                  "relative error between two consecutive "
                  "iterates is at \n  most %f" % (ftol, xtol), None],
              4: ["The cosine of the angle between func(x) and any "
                  "column of the\n  Jacobian is at most %f in "
                  "absolute value" % gtol, None],
              5: ["Number of calls to function has reached "
                  "maxfev = %d." % maxfev, ValueError],
              6: ["ftol=%f is too small, no further reduction "
                  "in the sum of squares\n  is possible.""" % ftol,
                  ValueError],
              7: ["xtol=%f is too small, no further improvement in "
                  "the approximate\n  solution is possible." % xtol,
                  ValueError],
              8: ["gtol=%f is too small, func(x) is orthogonal to the "
                  "columns of\n  the Jacobian to machine "
                  "precision." % gtol, ValueError],
              'unknown': ["Unknown error.", TypeError]}
    
    if maxfev is None:
        maxfev = 100 * (len(x0) + 1)

    def gtest(g):
        """test if the gradient has converged"""
        if gtol:
            return np.abs(g).max() <= gtol
        else:
            return False
    def xtest(dx, x):
        """see if the parameters have converged"""
        return la.norm(dx) <= xtol * (la.norm(x) + xtol)
    
    # set up update and chi2 for use
    if method == "ls":
        def update(x0, f):
            return _update_ls(x0, f, Dfun)
        
        def chi2(f):
            return _chi2_ls(f)

    elif method == "mle":
        def update(x0, f):
            return _update_mle(x0, f, Dfun)
        
        def chi2(f):
            return _chi2_mle(f)
    else:
        raise TypeError("Method {} not recognized".format(method))

    # get initial function, jacobian, hessian and gradient
    f = func(x0)
    j, a, g = update(x0, f)
    
    # initialize chi2
    chisq_old = chi2(f)
    
    # make our scaling factor
    mu = factor * np.diagonal(a).max()
    
    x = x0
    
    for ev in range(maxfev):
        if gtest(g):
            info = 4
            break
        # calculate proposed step
        aug_a = a + np.diag(np.ones_like(g) * mu)
        try:
            # https://software.intel.com/en-us/mkl-developer-reference-fortran-matrix-inversion-lapack-computational-routines
            # dx = -la.inv(aug_a) @ g
            dx = la.solve(aug_a, -g)
        except la.LinAlgError:
            mu *= factor
            continue
        if xtest(dx, x):
            info = 2
            break
        # make test move, I think I should be saving previous
        # position so that I can "undo" if this is bad
        x = x0 + dx
        f = func(x)
        chisq_new = chi2(f)
        # see if we reduced chisq, note we should do more here
        rho = chisq_old - chisq_new
        if rho > 0:
            # ftest
            if rho <= ftol * chisq_old:
                info = 1
                break
            # update params, chisq and a and g
            x0 = x
            chisq_old = chisq_new
            j, a, g = update(x0, f)
            mu /= factor
        else:
            mu *= factor
    else:
        # loop exited normally
        info = 5
            
    if method == "mle":
        # remember we return the data with f?
        f = f[0]

    infodict = dict(fvec=f, fjac=j, nfev=ev)
    
    if info not in [1, 2, 3, 4] and not full_output:
        if info in [5, 6, 7, 8]:
            warnings.warn(errors[info][0], RuntimeWarning)
        else:
            try:
                raise errors[info][1](errors[info][0])
            except KeyError:
                raise errors['unknown'][1](errors['unknown'][0])

    errmsg = errors[info][0]

    popt, cov_x = x, None
    
    if full_output:
        return popt, cov_x, infodict, errmsg, info
    else:
        return popt, cov_x


def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
                  check_finite=True, bounds=(-np.inf, np.inf), method=None,
                  jac=None, **kwargs):
    """
    Use non-linear least squares to fit a function, f, to data.
    Assumes ``ydata = poisson(f(xdata, *params))``
    Parameters
    ----------
    f : callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : An M-length sequence or an (k,M)-shaped array for functions with k predictors
        The independent variable where the data is measured.
    ydata : M-length sequence
        The dependent data --- nominally f(xdata, ...)
    p0 : None, scalar, or N-length sequence, optional
        Initial guess for the parameters.  If None, then the initial
        values will all be 1 (if the number of parameters for the function
        can be determined using introspection, otherwise a ValueError
        is raised).
    sigma : None or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:
            - A 1-d `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.
            - A 2-d `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.
              .. versionadded:: 0.19
        None (default) is equivalent of 1-d `sigma` filled with ones.
    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.
        If False, only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    check_finite : bool, optional
        If True, check that the input arrays do not contain nans of infs,
        and raise a ValueError if they do. Setting this parameter to
        False may silently produce nonsensical results if the input arrays
        do contain nans. Default is True.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each element of the tuple must be either an array with the length equal
        to the number of parameters, or a scalar (in which case the bound is
        taken to be the same for all parameters.) Use ``np.inf`` with an
        appropriate sign to disable bounds on all or some parameters.
        .. versionadded:: 0.17
    method : {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimization.  See `least_squares` for more details.
        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
        provided. The method 'lm' won't work when the number of observations
        is less than the number of variables, use 'trf' or 'dogbox' in this
        case.
        
        "ls", "mle"
        What type of estimator to use. Maximum likelihood ("mle") assumes that the noise
        in the measurement is poisson distributed while least squares ("ls") assumes
        normally distributed noise. "pyls" is a python implementation, for testing only
        
        .. versionadded:: 0.17
    jac : callable, string or None, optional
        Function with signature ``jac(x, ...)`` which computes the Jacobian
        matrix of the model function with respect to parameters as a dense
        array_like structure. It will be scaled according to provided `sigma`.
        If None (default), the Jacobian will be estimated numerically.
        String keywords for 'trf' and 'dogbox' methods can be used to select
        a finite difference scheme, see `least_squares`.
        .. versionadded:: 0.18
    kwargs
        Keyword arguments passed to `leastsq` for ``method='lm'`` or
        `least_squares` otherwise.""" 

    # fix kwargs
    return_full = kwargs.pop('full_output', False)
    can_full_output = method not in {'trf', 'dogbox'} and np.array_equal(bounds, (-np.inf, np.inf))

    if method in {'lm', 'trf', 'dogbox', None}:
        if can_full_output:
            kwargs['full_output'] = return_full
            
        res = scipy.optimize.curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma,
                  check_finite, bounds, method, jac, **kwargs)
        if can_full_output:
            return res
        else:
            return res[0], res[1], None, "No error", 1
    
    if method in {'ls', 'lm', 'trf', 'dogbox', None}:
        _wrap_func = _wrap_func_ls
        _wrap_jac = _wrap_jac_ls
    elif method == "mle":
        _wrap_func = _wrap_func_mle
        _wrap_jac = _wrap_jac_mle
    else:
        raise TypeError("Method {} not recognized".format(method))
    
    if p0 is None:
        raise NotImplementedError("You must give a guess")
    if sigma is not None:
        raise NotImplementedError("Weighting has not been implemented")
    else:
        transform = None
        
    if jac is None:
        raise NotImplementedError("You need a Jacobian")
        
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
            
    func = _wrap_func(f, xdata, ydata, transform)
    if callable(jac):
        jac = _wrap_jac(jac, xdata, transform)
    elif jac is None and method != 'lm':
        jac = '2-point'

    res = lm(func, p0, Dfun=jac, full_output=1, method=method, **kwargs)
    popt, pcov, infodict, errmsg, info = res
    cost = np.sum(infodict['fvec'] ** 2)


    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = la.svd(infodict['fjac'], full_matrices=False)
    threshold = np.finfo(float).eps * max(infodict['fjac'].shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)

    if info not in [1, 2, 3, 4]:
        raise RuntimeError("Optimal parameters not found: " + errmsg)

    if return_full:
        return popt, pcov, infodict, errmsg, info
    else:
        return popt, pcov