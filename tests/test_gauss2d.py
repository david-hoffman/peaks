from nose.tools import *
from skimage.external import tifffile as tif
from peaks.gauss2d import Gauss2D
import os
import numpy as np
import unittest

class TestGauss2D(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(64)
        self.y = np.arange(128)

        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.data = tif.imread(os.path.join(os.path.dirname(__file__),'..','fixtures','noisy_data.tif'))

        #read in test data, read in optimized fit params, read in optimized fit
        #everything should be in tiff file format.

    def test_data(self):
        '''
        Make sure object contains right data and that it returns a copy
        '''

        myg = Gauss2D(self.data)
        #check that data is the same
        assert np.all(self.data == myg.data)
        #but not the same object
        assert_is_not(self.data, myg.data)

    def test_estimate_params(self):
        myg = Gauss2D(self.data)
        assert np.allclose(myg.estimate_params(), np.array([  1.24887055e+00, \
            6.42375215e+01,   3.07687388e+01, 3.54327311e+01,   1.77964115e+01,\
            1.02789770e-02, 8.30703795e-01]))


    def test_rho(self):
        '''
        Test that rho is rejected if outside acceptable range
        '''

        coefs_full = [12,20,70,20,30,-1,2]
        assert_raises(ValueError, Gauss2D.gauss2D, (self.xx, self.yy),*coefs_full)

        coefs_full = [12,20,70,20,30,1,2]
        assert_raises(ValueError, Gauss2D.gauss2D, (self.xx, self.yy),*coefs_full)

    def test_unequal_range(self):
        '''
        Making sure that a ValueError is thrown for unequal ranges
        '''

        x = np.arange(64)
        y = np.arange(64)

        xx, yy = np.meshgrid(x, y)

        coefs_full = [12,20,70,20,30,0.5,2]

        assert_raises(ValueError, Gauss2D.gauss2D,(self.xx, yy),*coefs_full)
        assert_raises(ValueError, Gauss2D.gauss2D,(xx, self.yy),*coefs_full)

    def test_gauss2D_sym(self):
        '''
        Testing if the symmetrical case is a sub case of the full one
        '''

        coefs_full = [12,20,70,20,20,0,2]
        full_data = Gauss2D.gauss2D((self.xx, self.yy),*coefs_full)

        coefs_sym = coefs_full[:4]+[coefs_full[-1]]
        sym_data = Gauss2D.gauss2D_sym((self.xx, self.yy),*coefs_sym)

        assert np.all(sym_data == full_data)

    def test_gauss2D_norot(self):
        '''
        Testing if the no rotation case is a sub case of the full one
        '''

        coefs_full = [12,20,70,20,30,0,2]
        full_data = Gauss2D.gauss2D((self.xx, self.yy),*coefs_full)

        coefs_norot = coefs_full[:5]+[coefs_full[-1]]
        norot_data = Gauss2D.gauss2D_norot((self.xx, self.yy),*coefs_norot)

        assert np.all(norot_data == full_data)

    def test_model(self):
        '''
        Test model classmethod in Gauss2D class
        '''

        #first test acceptable values for goodness
        funcs = (Gauss2D.gauss2D,Gauss2D.gauss2D_norot,Gauss2D.gauss2D_sym)
        coefs = ([12,20,70,20,30,0.5,2], [12,20,70,20,30,2], [12,20,70,20,2])

        for func, coef in zip(funcs, coefs):
            data = func((self.xx, self.yy),*coef)
            model = Gauss2D.model((self.xx, self.yy),*coef)
            assert np.all(model == data)

        #now test edge cases
        assert_raises(ValueError, Gauss2D.model, 1)
        assert_raises(ValueError, Gauss2D.model, *np.arange(10))




# import matplotlib.pylab as plt
# import numpy as np
#
# from gauss2d import Gauss2D
#
# def plotdata(data, title):
#     fig, ax = plt.subplots()
#     ax.matshow(data)
#     ax.contour(data, 8, colors='k')
#     ax.set_title(title)
#     fig.show()
#     fig.tight_layout()
#
# def test_gauss2D():
#     #generate some shitty data
#     x = np.arange(64)
#     y = np.arange(128)
#
#     xx, yy = np.meshgrid(x, y)
#
#     coefs = [12,20,70,10,20,0.5,2]
#
#     data = Gauss2D.gauss2D((xx, yy),*coefs)
#
#     plotdata(data,'test_gauss2D')
#
# def test_gauss2D_norot():
#     #generate some shitty data
#     x = np.arange(64)
#     y = np.arange(128)
#
#     xx, yy = np.meshgrid(x, y)
#
#     coefs = (12,20,70,10,20,2)
#
#     data = Gauss2D.gauss2D_norot((xx, yy),*coefs)
#
#     plotdata(data,'test_gauss2D_norot')
#
# def test_gauss2D_norot_fit():
#     #generate some shitty data
#     x = np.arange(64)
#
#     y = np.arange(128)
#
#     xx, yy = np.meshgrid(x, y)
#
#     theta = 20*np.pi/180
#
#     xxr = (xx-30)*np.cos(theta)+(yy-90)*np.sin(theta)
#     yyr = (xx-30)*np.sin(theta)+(yy-90)*np.cos(theta)
#     data = 5+5*np.exp(-xxr**2/2/10**2-yyr**2/2/25**2)
#
#     noisy_data = Gauss2D(data+0.2*np.random.randn(*xx.shape))
#
#     guessparams = noisy_data.estimate_params()
#
#     noisy_data.optimize_params_ls(guessparams)
#
#     noisy_data_estimate = Gauss2D.model((xx,yy),*noisy_data.get_guess_params())
#
#     noisy_data_fit = Gauss2D.model((xx,yy),*noisy_data.get_opt_params())
#
#     fig, ax = plt.subplots()
#     ax.matshow(noisy_data._data)
#     ax.contour(noisy_data_estimate, 8, colors='r')
#     ax.contour(noisy_data_fit, 8, colors='k')
#     ax.set_title('Optimization Test')
#     fig.show()
#
# def test_gauss2D_sym():
#     #generate some shitty data
#     x = np.arange(64)
#     y = np.arange(128)
#
#     xx, yy = np.meshgrid(x, y)
#
#     coefs = (12,20,70,10,2)
#
#     data = Gauss2D.gauss2D_sym((xx, yy),*coefs)
#
#     plotdata(data,'test_gauss2D_sym')
#
# if __name__ == '__main__':
#     test_gauss2D()
#     test_gauss2D_norot()
#     test_gauss2D_sym()
#     test_gauss2D_norot_fit()
#
#     #blocking to show results
#     plt.show()
