
from sklearn import datasets
iris = datasets.load_iris()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit( iris.data, iris.target )

y_pred = gnb.predict( iris.data )

print( "%d muestras mal clasificadas de %d" % ( (iris.target != y_pred).sum(), iris.target.shape[0] ) )

print( "Accuracy = %.1f%%" % ( ( 100.0 * (iris.target == y_pred).sum() ) / iris.data.shape[0] ) )
