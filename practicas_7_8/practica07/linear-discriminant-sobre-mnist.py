
import numpy
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot

from machine_learning import MyKernelClassifier
from machine_learning import LinearDiscriminant

from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.mixture import GMM

print( "Loading ... " )

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

X = mnist.data
Y = mnist.target

X = X/255.0

print( "Transforming  ... " )

#X = PCA(n_components=150).fit_transform(X)
X = PCA(n_components=30).fit_transform(X)

#norm = StandardScaler()
norm = Normalizer()
X = norm.fit_transform(X)

X_test = X[60000:]
Y_test = Y[60000:]

X_train = X[:60000]
Y_train = Y[:60000]

#X_train,Y_train = shuffle( X_train, Y_train )
#X_train = X_train[:2000]
#Y_train = Y_train[:2000]

show_graphics = False
compare_with_kde = False

max_degree = 3
accuracy = numpy.zeros( [max_degree+1, 3 ] )

for degree in range( 1, max_degree+1 ):
    #temp_X_train = numpy.ones( [X_train.shape[0], 1+X_train.shape[1] ] )
    #temp_X_train[:,1:] = X_train[:,:]
    #temp_X_test = numpy.ones( [X_test.shape[0], 1+X_test.shape[1] ] )
    #temp_X_test[:,1:] = X_test[:,:]
    temp_X_train = X_train
    temp_X_test = X_test
    if degree >= 2:
        poly = PolynomialFeatures( degree = degree )
        temp_X_train = poly.fit_transform(temp_X_train)
        temp_X_test = poly.fit_transform(temp_X_test)

    print( "New dimensionality of samples %s" % str(temp_X_train.shape) )

    ld = LinearDiscriminant()
    ld.fit( temp_X_train, Y_train )

    y_pred = ld.predict( temp_X_train )
    train_accuracy = ( 100.0 * (Y_train == y_pred).sum() ) / len(Y_train)

    y_pred = ld.predict( temp_X_test )
    test_accuracy = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)

    print( "Linear discriminant classifier with polynomial transformation of degree %d " % degree )
    print( "%d missclassified samples from %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )
    print( "Accuracy = %.1f%%" % test_accuracy )

    accuracy[degree][0] = train_accuracy
    accuracy[degree][1] = test_accuracy

    if compare_with_kde :
        # For comparation purposes, we check the accuracy of a classifier based on Kernel Density Estimators
        # BEGIN
        mykc = MyKernelClassifier()
        mykc.fit( temp_X_train, Y_train )
        y_pred = mykc.predict( temp_X_test )
        test_accuracy_kde = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
        # END
        print( "Accuracy of the KDE classifier = %.1f%%" % test_accuracy_kde )
        accuracy[degree][2] = test_accuracy_kde
    print( " " )

if compare_with_kde :
    # For comparation purposes, we check the accuracy of a classifier based on Kernel Density Estimators without transforming the samples
    # BEGIN
    mykc = MyKernelClassifier()
    mykc.fit( X_train, Y_train )
    y_pred = mykc.predict( X_test )
    test_accuracy_kde = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
    # END
    print( "Accuracy of the KDE classifier = %.1f%%" % test_accuracy_kde )


if show_graphics:
    pyplot.title( "Evolution of Accuracy versus Degree " )
    pyplot.xlabel( "Degree" )
    pyplot.ylabel( "Accuracy" )
    artists=[]
    labels=[]
    x_range = numpy.arange(1,len(accuracy))
    plot1 = pyplot.plot( x_range, accuracy[1:,0], c='b' )
    artists.append( plot1[0] )
    labels.append( "Accuracy on training set " )
    plot2 = pyplot.plot( x_range, accuracy[1:,1], c='r' )
    artists.append( plot2[0] )
    labels.append( "Accuracy on test set " )

    if compare_with_kde :
        plot3 = pyplot.plot( x_range, accuracy[1:,2], c='g' )
        artists.append( plot3[0] )
        labels.append( "Accuracy on test set for KDE classifier" )
        pyplot.ylim( accuracy[1:,:].min()-10.0, 100.0 )
    else:
        pyplot.ylim( accuracy[1:,:2].min()-10.0, 100.0 )

    pyplot.legend( artists, labels, loc=4 )
    pyplot.grid()
    pyplot.show()
