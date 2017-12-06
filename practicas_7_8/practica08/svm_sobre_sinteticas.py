
import sys
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm

from machine_learning import MyKernelClassifier
from machine_learning import LinearDiscriminant
from machine_learning import generate_datasets

X_train,Y_train,X_test,Y_test = generate_datasets.generate_multivariate_normals( 5, 2, 150, 50, 5.0, 2.0 )

#
# Uncomment one of the following lines for normalizing or dimensionality reduction by means of PCA.
# Choose none, one of them or both.
# In case of applying both techniques decide the order at your convinience.
#
#norm = StandardScaler(); X = norm.fit_transform(X)
#X = PCA(n_components=30).fit_transform(X)


kernels = [ 'rbf', 'linear', 'poly', 'sigmoid' ]
degrees = [ 1, 2, 3, 4, 5 ]
gammas = [ 0.01, 0.1, 1.0, 2.0 ]
values_of_C = [ 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e+1, 1.0e+2, 1.0e+3 ]
coef0 = 1.0

for kernel in kernels:
    if kernel == 'poly' :
        _degrees = degrees
    else:
        _degrees = [1]
    for degree in _degrees:
        for gamma in gammas:
            for C in values_of_C:
                classifier = svm.SVC( kernel=kernel,
                                      degree=degree,
                                      gamma=gamma,
                                      coef0=coef0,
                                      C=C,
                                      max_iter=10000,
                                      verbose=False )
                classifier.fit( X_train, Y_train )
                y_pred = classifier.predict( X_test )

                accuracy = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
                print( " %-7s  degree %3d  gamma %.6f  C %e  Accuracy %.1f%%"
                           % (kernel, degree, gamma, C, accuracy ) )
                #print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )

# For comparation purposes, we check the accuracy of a classifier based on Kernel Density Estimators without transforming the samples
# BEGIN
mykc = MyKernelClassifier()
mykc.fit( X_train, Y_train )
y_pred = mykc.predict( X_test )
test_accuracy_kde = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
# END
print( "Accuracy of the KDE classifier = %.1f%%" % test_accuracy_kde )


# BEGIN
degree = 2
poly = PolynomialFeatures( degree = degree )
temp_X_train = poly.fit_transform(X_train)
temp_X_test = poly.fit_transform(X_test)
ld = LinearDiscriminant()
ld.fit( temp_X_train, Y_train )
y_pred = ld.predict( temp_X_test )
test_accuracy = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
print( "Linear discriminant classifier with polynomial transformation of degree %d " % degree )
#print( "%d missclassified samples from %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )
print( "Accuracy of the LD classifier = %.1f%%" % test_accuracy )

degree = 3
poly = PolynomialFeatures( degree = degree )
temp_X_train = poly.fit_transform(X_train)
temp_X_test = poly.fit_transform(X_test)
ld = LinearDiscriminant()
ld.fit( temp_X_train, Y_train )
y_pred = ld.predict( temp_X_test )
test_accuracy = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
print( "Linear discriminant classifier with polynomial transformation of degree %d " % degree )
#print( "%d missclassified samples from %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )
print( "Accuracy of the LD classifier = %.1f%%" % test_accuracy )
# END
