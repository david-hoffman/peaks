#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_gauss2d.py
"""
Test suite for `Gauss2D` class

Copyright (c) 2016, David Hoffman
"""

from nose.tools import *
import warnings
from skimage.external import tifffile as tif
from peaks.gauss2d import Gauss2D
import os
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import unittest
from itertools import product


class TestGauss2DBasics(unittest.TestCase):
    """Testing the Gauss2D function in peaks.gauss2d"""

    def setUp(self):
        # make x, y ranges
        self.x = np.arange(64)
        self.y = np.arange(128)
        # make meshgrid
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.noisy_data = tif.imread(
            os.path.join(os.path.dirname(__file__), "..", "fixtures", "noisy_data.tif")
        )
        self.raw_data = tif.imread(
            os.path.join(os.path.dirname(__file__), "..", "fixtures", "raw_data.tif")
        )
        self.myg = Gauss2D(self.noisy_data)

        # read in test data, read in optimized fit params, read in optimized
        # fit everything should be in tiff file format.

    def test_data(self):
        """Make sure object contains right data and that it returns a copy"""
        # check that data is the same
        assert_array_equal(self.noisy_data, self.myg.data)
        # but not the same object
        assert_is_not(self.noisy_data, self.myg.data)

    def test_estimate_params(self):
        """Test estimate params"""
        assert_allclose(
            self.myg.estimate_params(),
            np.array(
                [
                    1.24887055e00,
                    6.42375215e01,
                    3.07687388e01,
                    3.54327311e01,
                    1.77964115e01,
                    1.02789770e-02,
                    8.30703795e-01,
                ]
            ),
        )

    def test_fit(self):
        """Test fit of test data"""
        with warnings.catch_warnings():
            # we know we'll get a warning (see test below)
            warnings.simplefilter("ignore", UserWarning)
            self.myg.optimize_params(modeltype="full")

        assert_allclose(
            self.myg.opt_params,
            np.array(
                [
                    0.99788977,
                    70.01566965,
                    25.00790577,
                    19.99506797,
                    9.99125883,
                    0.49377357,
                    1.0001033,
                ]
            ),
            1e-5,
            1e-8,
        )

    def test_full_fit(self):
        """Make sure that when optimizing no warning is raised"""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            self.myg.optimize_params(modeltype="full")

    def test_gen_model(self):
        """Test model"""
        params = np.array([1, 70, 25, 20, 10, 0.5, 1])
        myg = Gauss2D.gen_model(self.noisy_data, *params)
        assert_allclose(self.raw_data, myg)

    def test_rho(self):
        """Test that rho is rejected if outside acceptable range"""
        coefs_full = (12, 20, 70, 20, 30, -1, 2)
        assert_warns(UserWarning, Gauss2D.model, (self.xx, self.yy), *coefs_full)

        coefs_full = (12, 20, 70, 20, 30, 1, 2)
        assert_warns(UserWarning, Gauss2D.model, (self.xx, self.yy), *coefs_full)

    def test_unequal_range(self):
        """Making sure that a ValueError is thrown for unequal ranges"""

        x = np.arange(64)
        y = np.arange(64)

        xx, yy = np.meshgrid(x, y)

        coefs_full = [12, 20, 70, 20, 30, 0.5, 2]

        assert_raises(RuntimeError, Gauss2D.gauss2D, (self.xx, yy), *coefs_full)
        assert_raises(RuntimeError, Gauss2D.gauss2D, (xx, self.yy), *coefs_full)

    def test_gauss2D_sym(self):
        """Testing if the symmetrical case is a sub case of the full one"""

        coefs_full = [12, 20, 70, 20, 20, 0, 2]
        full_data = Gauss2D.gauss2D((self.xx, self.yy), *coefs_full)

        coefs_sym = coefs_full[:4] + [coefs_full[-1]]
        sym_data = Gauss2D.gauss2D_sym((self.xx, self.yy), *coefs_sym)

        assert np.all(sym_data == full_data)

    def test_gauss2D_norot(self):
        """Testing if the no rotation case is a sub case of the full one"""

        coefs_full = [12, 20, 70, 20, 30, 0, 2]
        full_data = Gauss2D.gauss2D((self.xx, self.yy), *coefs_full)

        coefs_norot = coefs_full[:5] + [coefs_full[-1]]
        norot_data = Gauss2D.gauss2D_norot((self.xx, self.yy), *coefs_norot)

        assert np.all(norot_data == full_data)

    def test_model(self):
        """Test model classmethod in Gauss2D class"""

        # first test acceptable values for goodness
        funcs = (Gauss2D.gauss2D, Gauss2D.gauss2D_norot, Gauss2D.gauss2D_sym)
        coefs = ([12, 20, 70, 20, 30, 0.5, 2], [12, 20, 70, 20, 30, 2], [12, 20, 70, 20, 2])

        for func, coef in zip(funcs, coefs):
            data = func((self.xx, self.yy), *coef)
            model = Gauss2D.model((self.xx, self.yy), *coef)
            assert np.all(model == data)

        # now test edge cases
        assert_raises(ValueError, Gauss2D.model, 1)
        assert_raises(ValueError, Gauss2D.model, *np.arange(10))


class TestGauss2DArea(unittest.TestCase):
    """Test that area calculations are accurate"""

    def setUp(self):
        """Basic set up"""
        data = np.zeros((2001, 2001))
        self.gauss = Gauss2D(data)

    def test_sym(self):
        """Area test for the symmetric case"""
        self.gauss._popt = np.array((1, 1000, 1000, 100, 0))
        self._test()

    def test_norot(self):
        """Area test for the norotation case"""
        self.gauss._popt = np.array((1, 1000, 1000, 100, 75, 0))
        self._test()

    def test_full(self):
        """Area test for the full case"""
        self.gauss._popt = np.array((1, 1000, 1000, 100, 80, 0.5, 0))
        self._test()

    def _test(self):
        area = self.gauss.area()
        model = self.gauss.fit_model
        numeric_area = model.sum()
        assert_allclose(area, numeric_area, err_msg="Failed with GT params")
        assert_allclose(
            Gauss2D(model).area(guess_params=self.gauss._popt),
            numeric_area,
            err_msg="Failed when optimizing params",
        )


def _self_consistency_test_factory(guess, modeltype, fittype, snr):
    """Build the self-consistency tests so I don't have to"""
    # set up doc string for returned function
    doc_str = "Testing guess={}, modeltype={}, fittype={}, snr={}"
    # build name of function
    name = "Test_modeltype_{}_fittype_{}".format(modeltype, fittype)
    # build test (need the underscore otherwise nose will pick it up)...

    def _test(self):
        # pull proper data and coefs
        coefs = self.coefs[modeltype]
        data = self.data[modeltype]
        # add noise
        if snr:
            if fittype == "ls":
                amp = coefs[0]
                noise_data = (amp / snr) * np.random.randn(*data.shape)
                noisy_data = data + noise_data
                # this is a heuristic and should be replaced with a better
                # reasone parameter
                rtol = 1 / snr
            elif fittype == "mle":
                noisy_data = np.random.poisson(data)
                rtol = 1 / coefs[0]
        else:
            noisy_data = data
            rtol = 1e-7
        # build object to test
        test_g = Gauss2D(noisy_data)
        # speficify guesses
        if not guess:
            guess_coefs = None
        else:
            guess_coefs = coefs.copy()
        # grab optimized coefs
        test_coefs = test_g.optimize_params(
            guess_params=guess_coefs, modeltype=modeltype, fittype=fittype
        )
        # do the actual test
        assert_allclose(test_coefs, coefs, rtol=rtol)

    # add doc string
    _test.__doc__ = doc_str.format(guess, modeltype, fittype, snr)
    if snr:
        name += "_wSNR_{}".format(snr)
    if guess:
        name += "_wGuess"
    _test.__name__ = name
    return _test


class BuildTestsMeta(type):
    """A meta class to build our tests for us"""

    def __new__(mcs, name, bases, dictionary):
        # loop through the product of different inputs
        for guess, modeltype, fittype, snr in product(
            (True,), ("sym", "norot", "full"), ("ls", "mle"), (0, 10)
        ):
            # build the actual test
            _test = _self_consistency_test_factory(guess, modeltype, fittype, snr)
            # update the test's __doc__ so that different
            # classes will have unique names
            _test.__doc__ = name + " " + _test.__doc__
            dictionary[_test.__name__] = _test
        # build class
        return type.__new__(mcs, name, bases, dictionary)


class _TestGauss2DSelfConsistencyBase(unittest.TestCase, metaclass=BuildTestsMeta):
    """Base class for self consistency testing"""

    def _make_coefs(self):
        """A function to be implemented by the child classes to generate coefs"""
        raise NotImplementedError

    def setUp(self):
        """Set up testing fixtures, they'll be generated on the fly because
        this is a small model"""
        amp, x0, y0, sigma_x, sigma_y, rho, offset = self._make_coefs()
        # make full
        gt_coefs_full = np.array((amp, x0, y0, sigma_x, sigma_y, rho, offset))
        gt_full = Gauss2D.model((self.xx, self.yy), *gt_coefs_full)
        # make norot
        gt_coefs_norot = np.array((amp, x0, y0, sigma_x, sigma_y, offset))
        gt_norot = Gauss2D.model((self.xx, self.yy), *gt_coefs_norot)
        # make sym
        gt_coefs_sym = np.array((amp, x0, y0, sigma_x, offset))
        gt_sym = Gauss2D.model((self.xx, self.yy), *gt_coefs_sym)
        # make internal dictionaries for tests to use.
        self.coefs = dict(sym=gt_coefs_sym, norot=gt_coefs_norot, full=gt_coefs_full)
        self.data = dict(sym=gt_sym, norot=gt_norot, full=gt_full)


class TestGauss2DSelfConsistencyFixed(_TestGauss2DSelfConsistencyBase):
    """Test self consistency with fixed numbers"""

    test_str = "Fixed test"

    def _make_coefs(self):
        """Fixed Coefs"""
        ny, nx = 16, 16
        # make grid
        self.yy, self.xx = np.indices((ny, nx))
        # choose center
        y0, x0 = 5, 13
        # choose sigmas
        sigma_y, sigma_x = 5, 3
        # choose amp
        amp = 20
        # choose rho
        rho = 0.5
        # choose offset
        offset = 10
        return amp, x0, y0, sigma_x, sigma_y, rho, offset


class TestGauss2DSelfConsistencyRand(_TestGauss2DSelfConsistencyBase):
    """Test self consistency with random numbers"""

    test_str = "Random test"

    def _make_coefs(self):
        """"""
        ny, nx = np.random.randint(10, 100, 2)
        # make grid
        self.yy, self.xx = np.indices((ny, nx))
        # choose center
        y0, x0 = (np.random.random(2) * 0.9 + 0.10) * (ny, nx)
        # choose sigmas
        sigma_y, sigma_x = (0.1 * np.random.random(2) + 0.1) * (ny, nx)
        # choose amp
        amp = np.random.random() * 100 + 10
        # choose rho
        rho = (np.random.random() * 2 - 1) * 0.99
        # choose offset
        offset = np.random.random() * 100
        return amp, x0, y0, sigma_x, sigma_y, rho, offset
