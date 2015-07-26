import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def detrend_data(data, degree = 1):
    '''
    Take 2D (i.e. image) data and remove the background using a polynomial fit

    Eventually this will be generalized to data of any dimension and perhaps

    Parameters
    ----------
    data : ndarray (NxM)
        data to detrend
    degree : int
        the degree of the polynomial with which to model the background

    Returns
    -------
    out : ndarray (NxM)
        background
    '''

    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])

    xx, yy = np.meshgrid(x,y)

    #We have to take our 2D data and transform it into a list of 2D coordinates
    X = np.dstack((xx.ravel(),yy.ravel())).reshape((np.prod(data.shape),2))

    #We have to ravel our data so that it is a list of points
    vector = data.ravel()

    #now we can continue as before
    predict= X
    poly = PolynomialFeatures(degree)
    X_ = poly.fit_transform(X)
    predict_ = poly.fit_transform(predict)
    clf = linear_model.RANSACRegressor()
    clf.fit(X_, vector)

    #we have to reshape our fit to mirror our original data
    background = clf.predict(predict_).reshape(data.shape)

    return background
