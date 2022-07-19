#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_utils.py
"""
Test suite for `utils.py` of the peaks package

Copyright (c) 2017, David Hoffman
"""

from peaks.utils import *
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import unittest


def fixed_signs(popt1, popt2):
    """Fix signs of variables for comparison."""
    if np.sign(popt1[0]) != np.sign(popt2[0]):
        print("Changing signs ...")
        if popt2[2] < 0:
            popt2[2] += np.pi
        else:
            popt2[2] -= np.pi
        popt2[0] *= -1
    return popt1, popt2


class TestSineFit(unittest.TestCase):
    """A test case for the PeakFinder class"""

    def setUp(self):
        """Set up our variables"""
        shape = np.random.randint(5, 50)
        periods = self.periods = np.random.rand() * 3 + 1
        freq = periods / shape
        amp = np.random.randn()
        offset = np.random.randn()
        phase = (np.random.rand() - 1 / 2) * 5 / 3 * np.pi
        p_gt = self.p_gt = (amp, freq, phase, offset)
        x = self.x = np.arange(shape)
        self.data = sine(x, *p_gt)

    def test_self_consistency_no_noise(self):
        """See if we can fit noiseless data right"""
        print(np.array(self.p_gt))
        popt, pcov = sine_fit(self.data, self.periods)
        print(popt)
        assert_allclose(*fixed_signs(self.p_gt, popt), 1e-4)

    def test_self_consistency_noise(self):
        """See if we can fit noiseless data right"""
        # test with SNR = 100
        print(np.array(self.p_gt))
        SNR = self.p_gt[0] / 100
        noisy_data = self.data + SNR * np.random.randn(*self.data.shape)
        popt, pcov = sine_fit(noisy_data, self.periods)
        print(popt)
        assert_allclose(*fixed_signs(self.p_gt, popt), 5e-1)
