
import numpy
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import GaussianNB

print( "Loading ... " )

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

print( "Transforming  ... " )

X_test = mnist.data[60000:]
Y_test = mnist.target[60000:]

(X_train,Y_train) = shuffle( mnist.data[:60000], mnist.target[:60000] )


gnb = GaussianNB()
gnb.fit( X_train, Y_train )

print( "Predicting  ... " )

y_pred = gnb.predict( X_test )

print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )

print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )
