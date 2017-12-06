
import numpy

from sklearn.datasets import fetch_mldata
from matplotlib import pyplot

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

print( mnist.data.shape )
print( numpy.unique( mnist.target ) )


n = numpy.random.randint( len(mnist.data) );

sample = mnist.data[n].reshape(28,28)

pyplot.imshow( sample, cmap=pyplot.cm.gray_r, interpolation=None )
pyplot.show();
