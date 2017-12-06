
import sys
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

X = mnist.data
Y = mnist.target

Y=Y.astype(numpy.int64)

#
# Uncomment one of the following lines for normalizing or dimensionality reduction by means of PCA.
# Choose none, one of them or both.
# In case of applying both techniques decide the order at your convinience.
#
print( "Normalizing ... " )
#norm = Normalizer(); X = norm.fit_transform(X)
X = X/255.0
#norm = StandardScaler(); X = norm.fit_transform(X)
print( "Applying PCA ... " )
X = PCA(n_components=30).fit_transform(X)
#X = StandardScaler().fit_transform(X)


# Separate test set and training set
X_test = X[60000:]
Y_test = Y[60000:]

X_train = X[:60000]
Y_train = Y[:60000]


kernels = [ 'rbf', 'linear', 'poly', 'sigmoid' ]
degrees = [ 1, 2, 3, 4, 5 ]
gammas = [ 0.001 ]
values_of_C = [ 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e+1, 1.0e+2, 1.0e+3 ]
coef0 = 1.0

print( "Testing different combinations of kernels and values of different parameters ... " )
for kernel in kernels:
    if kernel == 'poly' :
        _degrees = degrees
    else:
        _degrees = [1]
    for degree in _degrees:
        for gamma in gammas:
            for C in values_of_C:
                classifier = svm.SVC( kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C, max_iter=10000, verbose=False )
                classifier.fit( X_train, Y_train )
                y_pred = classifier.predict( X_test )

                accuracy = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
                print( " %-7s  degree %3d  gamma %.6f  C %e  Accuracy %.1f%%" % (kernel, degree, gamma, C, accuracy ) )
