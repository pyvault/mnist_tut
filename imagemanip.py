from opendeep.data.dataset import TEST
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.utils.noise import salt_and_pepper
import matplotlib.pyplot as plt
import numpy as np
from opendeep.data.dataset import MemoryDataset
import theano.tensor as T
import random
import PIL
import numpy
from custom_noise import salt_and_pepper_custom
from opendeep.utils.image import tile_raster_images

mnist = MNIST()
test_data, _ = mnist.getSubset(TEST)
test_data = test_data[:25].eval()
print test_data.shape

"""
(nrow, ncol) = test_data.shape
test_data_md = np.ndarray(shape=(nrow, ncol), dtype=object)
for i in range(nrow):
    test_data_md[i, :] = salt_and_pepper_custom(test_data[i, :])

print test_data_md.shape
"""







"""
saltarr = np.ones((28, 28))
saltarr2 = np.ones((28, 28))

dim = 28
pixs = 3
range_lim = 1
range_lim2 = 1

rdarray_x = random.sample(range(0, dim - pixs), range_lim)
rdarray_x2 = random.sample(range(0, dim - pixs), range_lim)

rdarray_y = random.sample(range(0, dim - pixs), range_lim2)
rdarray_y2 = random.sample(range(0, dim - pixs), range_lim2)

for x in range(0, range_lim):
    saltarr[rdarray_x[x]:rdarray_x[x] + pixs, rdarray_x2[x]:rdarray_x2[x] + pixs] = 0
for x in range(0, range_lim2):
    saltarr2[rdarray_y[x]:rdarray_y[x] + pixs, rdarray_y2[x]:rdarray_y2[x] + pixs] = 1

saltzero = np.logical_not(saltarr) * saltarr2

plt.imshow(np.logical_not(saltarr), cmap='Greys_r')
plt.show()

"""

"""
mnist = MNIST()
print type(mnist)
K = salt_and_pepper(input=mnist, noise_level=0.4)

test_data, _ = mnist.getSubset(TEST)
test_data = test_data[:25].eval()

print 'Test data'
print type(test_data)
print test_data.shape

test_data_md = np.ndarray(shape=(2, 784))

test_data_s = test_data[1, :]
test_data_s2 = test_data[2, :]

print 'Test data s'
print type(test_data_s)
print test_data_s.shape

test_data_srs = np.reshape(test_data_s, (28, 28))
#test_data_srs[5:15, 5:15] = 1
#test_data_srs_alt = np.reshape(test_data_srs, 784)

test_data_srs2 = np.reshape(test_data_s2, (28, 28))
#test_data_srs2[5:15, 5:15] = 1
#test_data_srs2_alt = np.reshape

ts_stack = np.hstack([test_data_srs, test_data_srs2])
plt.imshow(ts_stack, cmap='Greys_r')
plt.show()
"""




"""
dim = 28
pixs = 2
range_lim = 20

rdarray = random.sample(range(0, dim-pixs), range_lim)
rdarray2 = random.sample(range(0, dim-pixs), range_lim)

print rdarray
print rdarray2

for x in range(0, range_lim):
        test_data_srs[rdarray[x]:rdarray[x]+pixs, rdarray2[x]:rdarray2[x]+pixs] = 1

for x in range(0, range_lim):
        test_data_srs2[rdarray[x]:rdarray[x] + pixs, rdarray2[x]:rdarray2[x] + pixs] = 1

stacked = np.hstack([test_data_srs, test_data_srs2, test_data_srs])
d_stacked = np.hstack([stacked, stacked])

plt.imshow(d_stacked, cmap='Greys_r')
plt.show()
"""

"""
print 'Test data srs'
print type(test_data_srs)
print test_data_srs.shape

test_data_md[0, :] = test_data_srs_alt
test_data_md[1, :] = test_data_srs2_alt
test_data_md = test_data_md.astype(np.float32)

print 'Test data md'
print type(test_data_md)
print test_data_md.shape
"""



"""
corrupted_test = salt_and_pepper(test_data_md, 0.4).eval()

'Corrupted test'
print type(corrupted_test)
print corrupted_test.shape

corrupted_pick = corrupted_test[1, :]
corrupted_pick_f = np.reshape(corrupted_pick, (28, 28))

plt.imshow(corrupted_pick_f, cmap='Greys_r')
plt.show()
"""

"""
test_arr = np.ndarray

test_img = test_data[1, :]
test_img_rs = np.reshape(test_img, (28, 28))
test_img_rs[5:15, 5:15] = 1
test_img_rs_b = np.reshape(test_img_rs, 784)
test_arr[1] = test_img_rs_b
test_arr[2] = test_img_rs_b
"""

#data = MemoryDataset(train_X=test_img, test_X=test_img_rs_b)

"""
test_img2 = data.getSubset(2)

test_img3 = np.reshape(test_img2, (28, 28))

plt.imshow(test_img3)
plt.show()
"""

"""
corrupted_test = salt_and_pepper(test_img, 0.1).eval()
corrupted_test_rs = np.reshape(corrupted_test, (28, 28))
plt.imshow(corrupted_test_rs, cmap='Greys_r')
plt.show()
"""


"""
print test_img.shape
test_img_alt = test_data[2, :]

test_img2 = np.reshape(test_img, (28, 28))
print test_img2.shape

test_img_alt2 = np.reshape(test_img_alt, (28, 28))
#print test_img2.shape

#plt.imshow(test_img2[1:15, 1:15], cmap='Greys_r')
#plt.show()
#print test_img2

test_img2[5:15, 5:15] = 1

test_img3 = np.hstack([test_img_alt2, test_img2])

plt.imshow(test_img3, cmap='Greys_r')
plt.show()

tmat = T.dmatrix('X')
tmat = test_img3
print type(tmat)
#test_img3 = test_img2.reshape((test_img2.shape[0], -1))
#test_img3 = np.reshape(test_img2, 784)
#print test_img3.shape
"""
