from theano import function
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import theano.compat.six as six
from theano import shared
import numpy as np
import theano
import random


def salt_and_pepper_custom(input):

    #random.seed(23455)

    saltarr = np.ones((28, 28)) #ones
    saltarr2 = np.zeros((28, 28)) #zeros

    dim = 28
    pixs = 5
    range_lim = 1 #white
    range_lim2 = 0 #black

    rdarray_x = random.sample(range(0, dim - pixs), range_lim)
    rdarray_x2 = random.sample(range(10, 20), range_lim)

    #rdarray_y = random.sample(range(0, dim - pixs), range_lim2)
    #rdarray_y2 = random.sample(range(0, dim - pixs), range_lim2)
    #git test
    for x in range(0, range_lim):
        saltarr[rdarray_x[x]:rdarray_x[x] + pixs, rdarray_x2[x]:rdarray_x2[x] + pixs] = 0 #0
    for y in range(0, range_lim):
        saltarr2[rdarray_x[y]:rdarray_x[y] + pixs, rdarray_x2[y]:rdarray_x2[y] + pixs] = 1 #1
    """
    for z in range(0, range_lim2):
        saltarr[rdarray_y[z]:rdarray_y[z] + pixs, rdarray_y2[z]:rdarray_y2[z] + pixs] = 0  # 1
    """
    saltarr = np.reshape(saltarr, 784)
    saltarr2 = np.reshape(saltarr2, 784)
    saltzero = np.logical_not(saltarr) * saltarr2
    """
    saltzero = np.reshape(saltzero, (28, 28))
    for z in range(0, range_lim2):
        saltzero[rdarray_y[z]:rdarray_y[z] + pixs, rdarray_y2[z]:rdarray_y2[z] + pixs] = 0  # 1

    saltzero = np.reshape(saltzero, 784)
    """
    saltarr_c = theano.shared(saltarr.astype(np.float32))
    saltzero = theano.shared(saltzero.astype(np.float32))
    return input * saltarr_c + saltzero

