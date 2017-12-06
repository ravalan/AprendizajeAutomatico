
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

N=100

x1 = numpy.random.multivariate_normal( [  10.0, -16.0,  10.0 ], [ [  7.0, -4.0,  2.0],
                                                                  [ -4.0,  8.0,  0.0],
                                                                  [  2.0,  0.0,  5.0] ],  N )

x2 = numpy.random.multivariate_normal( [ -10.0, -15.0,  30.0 ], [ [ 17.0,  10.0,   0.0],
                                                                  [ 10.0,  13.0,   3.0],
                                                                  [  0.0,   3.0,  15.0] ],  N )

x3 = numpy.random.multivariate_normal( [  15.0,   0.0,  -8.0 ], [ [  5.0,   6.0,   5.0],
                                                                  [  6.0,  11.0,   4.0],
                                                                  [  5.0,   4.0,   9.0] ],  N )

print( x1.shape )

fig = pyplot.figure( 1, figsize=(8, 6) )
ax = Axes3D( fig, elev=-150, azim=110 )

ax.scatter( x1[:,0], x1[:,1], x1[:,2], c='r' ) #c=1, cmap=pyplot.cm.Paired )
ax.scatter( x2[:,0], x2[:,1], x2[:,2], c='b' ) #c=2, cmap=pyplot.cm.Paired )
ax.scatter( x3[:,0], x3[:,1], x3[:,2], c='g' ) #c=3, cmap=pyplot.cm.Paired )

ax.set_title( "Three Gaussian distributions in 3D" )
ax.set_xlabel( "1st dimension" )
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel( "2nd dimension" )
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel( "3rd dimension" )
ax.w_zaxis.set_ticklabels([])

pyplot.show()
