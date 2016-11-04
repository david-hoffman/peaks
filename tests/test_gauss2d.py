from nose.tools import *
import warnings
from skimage.external import tifffile as tif
from peaks.gauss2d import Gauss2D
import os
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import unittest


class TestGauss2DBasics(unittest.TestCase):
    """Testing the Gauss2D function in peaks.gauss2d"""

    def setUp(self):
        # make x, y ranges
        self.x = np.arange(64)
        self.y = np.arange(128)
        # make meshgrid
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.noisy_data = tif.imread(os.path.join(os.path.dirname(__file__),
                                                  '..', 'fixtures',
                                                  'noisy_data.tif'))
        self.raw_data = tif.imread(os.path.join(os.path.dirname(__file__),
                                                '..', 'fixtures',
                                                'raw_data.tif'))
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
        assert_allclose(self.myg.estimate_params(),
                        np.array([1.24887055e+00,
                                  6.42375215e+01,
                                  3.07687388e+01,
                                  3.54327311e+01,
                                  1.77964115e+01,
                                  1.02789770e-02,
                                  8.30703795e-01]))

    def test_fit(self):
        """Test fit of test data"""
        with warnings.catch_warnings():
            # we know we'll get a warning (see test below)
            warnings.simplefilter("ignore", UserWarning)
            self.myg.optimize_params(modeltype='full')

        assert_allclose(self.myg.opt_params,
                        np.array([0.99788977,
                                  70.01566965,
                                  25.00790577,
                                  19.99506797,
                                  9.99125883,
                                  0.49377357,
                                  1.0001033]),
                        1e-5, 1e-8)

    def test_gen_model(self):
        """Test model"""
        params = np.array([1,
                           70,
                           25,
                           20,
                           10,
                           0.5,
                           1])
        myg = Gauss2D.gen_model(self.noisy_data, *params)
        assert_allclose(self.raw_data, myg)

    def test_rho(self):
        """Test that rho is rejected if outside acceptable range"""
        coefs_full = [12, 20, 70, 20, 30, -1, 2]
        assert_warns(UserWarning, Gauss2D.gauss2D, (self.xx, self.yy),
                     *coefs_full)

        coefs_full = [12, 20, 70, 20, 30, 1, 2]
        assert_warns(UserWarning, Gauss2D.gauss2D, (self.xx, self.yy),
                     *coefs_full)

    def test_unequal_range(self):
        """Making sure that a ValueError is thrown for unequal ranges"""

        x = np.arange(64)
        y = np.arange(64)

        xx, yy = np.meshgrid(x, y)

        coefs_full = [12, 20, 70, 20, 30, 0.5, 2]

        assert_raises(ValueError, Gauss2D.gauss2D, (self.xx, yy), *coefs_full)
        assert_raises(ValueError, Gauss2D.gauss2D, (xx, self.yy), *coefs_full)

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
        coefs = ([12, 20, 70, 20, 30, 0.5, 2],
                 [12, 20, 70, 20, 30, 2],
                 [12, 20, 70, 20, 2])

        for func, coef in zip(funcs, coefs):
            data = func((self.xx, self.yy), *coef)
            model = Gauss2D.model((self.xx, self.yy), *coef)
            assert np.all(model == data)

        # now test edge cases
        assert_raises(ValueError, Gauss2D.model, 1)
        assert_raises(ValueError, Gauss2D.model, *np.arange(10))


class TestGauss2DSelfConsistency(unittest.TestCase):
    """test the self consistency of the model"""

    def setUp(self):
        """Set up our parameters"""
        # choose size
        ny, nx = np.random.randint(10, 100, 2)
        # make grid
        self.yy, self.xx = np.indices((ny, nx))
        # choose center
        y0, x0 = np.random.random(2) * (ny, nx)
        # choose sigmas
        sigma_y, sigma_x = 0.1 * np.random.random(2) * (ny, nx)
        # choose amp
        amp = (np.random.random() + 1.0) * 10
        # choose rho
        rho = (np.random.random() * 2 - 1) * 0.99
        # choose offset
        offset = np.random.random() * 10
        self.gt_coefs_full = np.array((amp, x0, y0, sigma_x, sigma_y, rho,
                                       offset))
        self.gt_full = Gauss2D.model((self.xx, self.yy),
                                     *self.gt_coefs_full)
        self.gt_coefs_norot = np.array((amp, x0, y0, sigma_x, sigma_y, offset))
        self.gt_norot = Gauss2D.model((self.xx, self.yy),
                                      *self.gt_coefs_norot)
        self.gt_coefs_sym = np.array((amp, x0, y0, sigma_x, offset))
        self.gt_sym = Gauss2D.model((self.xx, self.yy),
                                    *self.gt_coefs_sym)

    def test_setup(self):
        """test that setup works"""
        pass

    def _test_fit_no_noise(ground_truth, ground_truth_coefs, modeltype,
                           fittype):
        """Test full fit"""
        test_g = Gauss2D(ground_truth)
        test_coefs = test_g.optimize_params(modeltype=modeltype,
                                            fittype=fittype)
        assert_allclose(test_coefs, ground_truth_coefs)

    def test_fit_norot_no_noise_no_guess(self):
        """Make sure results are good for norot, no noise, no guess"""
        test_g = Gauss2D(self.gt_norot)
        test_coefs = test_g.optimize_params(modeltype='norot', fittype='ls')
        assert_allclose(test_coefs, self.gt_coefs_norot)

    def test_fit_norot_no_noise(self):
        """Make sure results are good for norot, no noise"""
        test_g = Gauss2D(self.gt_norot)
        test_coefs = test_g.optimize_params(self.gt_coefs_norot, fittype='ls')
        assert_allclose(test_coefs, self.gt_coefs_norot)

    def test_fit_norot_no_noise_no_guess_mle(self):
        """Make sure results are good for norot, no noise, no guess, mle"""
        test_g = Gauss2D(self.gt_norot)
        test_coefs = test_g.optimize_params(modeltype='norot', fittype='mle')
        assert_allclose(test_coefs, self.gt_coefs_norot)

    def test_fit_norot_no_noise_mle(self):
        """Make sure results are good for norot, no noise, mle"""
        test_g = Gauss2D(self.gt_norot)
        test_coefs = test_g.optimize_params(self.gt_coefs_norot, fittype='mle')
        assert_allclose(test_coefs, self.gt_coefs_norot)



























