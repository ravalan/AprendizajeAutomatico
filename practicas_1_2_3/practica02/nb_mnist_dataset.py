
from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import GaussianNB

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

gnb = GaussianNB()

gnb.fit( mnist.data, mnist.target )

y_pred = gnb.predict( mnist.data )

print( "%d muestras mal clasificadas de %d" % ( (mnist.target != y_pred).sum(), mnist.data.shape[0] ) )

print( "Accuracy = %.1f%%" % ( ( 100.0 * (mnist.target == y_pred).sum() ) / mnist.data.shape[0] ) )
