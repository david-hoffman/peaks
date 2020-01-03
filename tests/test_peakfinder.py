#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_peakfinder.py
"""
Test suite for `PeakFinder` class

Copyright (c) 2016, David Hoffman
"""

from nose.tools import *
from peaks.peakfinder import PeakFinder
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import unittest


class TestPeakFinder(unittest.TestCase):
    """A test case for the PeakFinder class"""

    def setUp(self):
        """Set up our variables"""
        shape = (256, 512)
        data = np.zeros(shape)
        points = self.points = (np.random.rand(10, 2) * shape).astype(int)
        data[points.T[0], points.T[1]] = 1
        assert data.sum() == 10, "Something wrong with data generation," " points = {}".format(
            points
        )
        self.data = data

    def test_self_consistency(self):
        pf = PeakFinder(self.data)
        pf.thresh = 0.1
        pf.find_blobs()
        found_points = np.sort(pf.blobs[:, :2], 0)
        assert_array_equal(found_points, np.sort(self.points, 0))
