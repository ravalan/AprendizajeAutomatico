
import numpy
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import GaussianNB

print( "Loading ... " )

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

print( "Transforming  ... " )

X = mnist.data
Y = mnist.target
new_X = numpy.ndarray( [ len(X), 56 ] )
#
#for n in range(len(X)):
#    sample = X[n].reshape(28,28)
#    for i in range(28): new_X[n,i] = sample[i,:].sum()
#    for j in range(28): new_X[n,28+j] = sample[:,j].sum()
#    new_X[n] = new_X[n] / new_X[n].sum()
#
X2=X.reshape( len(X), 28, 28 )
new_X[:,:28] = X2.sum(axis=1)
new_X[:,28:] = X2.sum(axis=2)
new_X[:,:] = new_X[:,:] / new_X.sum(axis=1)[:,numpy.newaxis]

print( new_X[0] )

X_test = new_X[60000:]
Y_test = Y[60000:]

print( "Shuffling  ... " )

(X_train,Y_train) = shuffle( new_X[:60000], Y[:60000] )


print( "Learning  ... " )

gnb = GaussianNB()
gnb.fit( X_train, Y_train )

print( "Predicting  ... " )

y_pred = gnb.predict( X_test )

print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )

print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )
