
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from machine_learning import MyKernel
from machine_learning import MyKernelClassifier
from sklearn.utils import shuffle

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

X = mnist.data
Y = mnist.target

#
# Uncomment one of the following lines for normalizing or dimensionality reduction by means of PCA.
# Choose none, one of them or both.
# In case of applying both techniques decide the order at your convinience.
#
#norm = Normalizer(); X = norm.fit_transform(X)

# Separate test set and training set
X_test = X[60000:]
Y_test = Y[60000:]

X_train = X[:60000]
Y_train = Y[:60000]

(X_train,Y_train) = shuffle( X_train, Y_train )

# Uncomment these two lines for working with a subset
X_train = X_train[:10000]
Y_train = Y_train[:10000]

pca = PCA(n_components=30)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

print( "PCA completed!" )

# BEGIN
mykc = MyKernelClassifier()
mykc.fit( X_train, Y_train )

print( "FIT completed!" )

y_pred = mykc.predict( X_test )


test_accuracy_kde = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
# END
print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )
print( "Accuracy of the KDE classifier = %.1f%%" % test_accuracy_kde )
