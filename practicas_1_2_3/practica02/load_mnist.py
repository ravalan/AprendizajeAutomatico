
import numpy

from sklearn.datasets import fetch_mldata


# --------------------------------------------------------------------------------
def save_to_file( filename, X, Y ):
    #
    N=len(X)
    dimX=X.shape[1]
    dimY=0
    if len(Y.shape) > 1: dimY = Y.shape[1]
    #
    of = open( filename, 'w' )
    of.write( "%d %d %d \n" % ( N, dimX, dimY if dimY > 0 else 1 ) )
    n=0
    while n < N:
        for i in range(dimX): of.write( "%f " % X[n,i] )
        if dimY == 0 :
            of.write( " %f\n" % Y[n] )
        else:
            for i in range(dimY): of.write( " %f" % Y[n,i] )
            of.write( "\n" )
        n=n+1
    of.close()
# --------------------------------------------------------------------------------

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

print( mnist.data.shape )
print( type(mnist.data) )
print( len(mnist.data.shape) )
print( mnist.target.shape )
print( type(mnist.target) )
print( len(mnist.target.shape) )

print( numpy.unique( mnist.target ) )

#for i in xrange(len(mnist.target)): print "%5d %.0f" % (i, mnist.target[i])

save_to_file( "mnist-training.txt", mnist.data[:60000 ], mnist.target[:60000 ] )
save_to_file( "mnist-test.txt",     mnist.data[ 60000:], mnist.target[ 60000:] )

