from PIL import Image
import numpy
import matplotlib.pyplot as plt
import glob
from opendeep.data.dataset import MemoryDataset

import theano.tensor as T


imageFolderPath = 'eyes/'
imagePath = glob.glob(imageFolderPath+'/*.JPG')

im_array = numpy.array( [numpy.array(Image.open(imagePath[i]).convert('L'), 'f') for i in range(len(imagePath))] )
flatten_array = im_array.reshape((im_array.shape[0], -1))
#print flatten_array[190]

#plt.imshow(flatten_array[5], cmap='Greys_r')
#plt.show()

data = MemoryDataset(train_X=flatten_array[1:180], test_X=flatten_array[180:191])

print data.getSubset(2)