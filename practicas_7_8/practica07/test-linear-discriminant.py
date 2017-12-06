
import sys
import numpy
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot

from machine_learning import MyKernelClassifier
from machine_learning import LinearDiscriminant
from machine_learning import generate_datasets

if __name__ == "__main__" :

    show_graphics=True
    compare_with_kde = True

    X_train,Y_train,X_test,Y_test = generate_datasets.generate_multivariate_normals( 5, 2, 150, 50, 5.0, 2.0 )

    if show_graphics: 
        #fig,ax = pyplot.subplots( 1, 1, sharey=False )
        pyplot.scatter( X_train[:,0], X_train[:,1], c=Y_train, s=50, edgecolors='none' )
        pyplot.show()

    max_degree = 10
    accuracy = numpy.zeros( [max_degree, 3 ] )
    best_ldc_accuracy = 0.0
    best_kde_accuracy = 0.0

    for degree in range( 1, max_degree ):
        poly = PolynomialFeatures( degree = degree )
        temp_X_train = poly.fit_transform(X_train)
        temp_X_test  = poly.fit_transform(X_test)

        print( "New dimensionality of samples %s" % str(temp_X_train.shape) )

        ld = LinearDiscriminant( l2_penalty=0.1 )
        ld.fit( temp_X_train, Y_train )

        y_pred = ld.predict( temp_X_train )
        train_accuracy = ( 100.0 * (Y_train == y_pred).sum() ) / len(Y_train)

        y_pred = ld.predict( temp_X_test )
        test_accuracy = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)

        best_ldc_accuracy = max( best_ldc_accuracy, test_accuracy )
    
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
            best_kde_accuracy = max( best_kde_accuracy, test_accuracy_kde )
        print( " " )


    print( "Best accuracy of the linear discriminant classifier = %.1f%%" % best_ldc_accuracy )

    if compare_with_kde :
        # For comparation purposes, we check the accuracy of a classifier based on Kernel Density Estimators without transforming the samples
        # BEGIN
        mykc = MyKernelClassifier()
        mykc.fit( X_train, Y_train )
        y_pred = mykc.predict( X_test )
        test_accuracy_kde = ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test)
        best_kde_accuracy = max( best_kde_accuracy, test_accuracy_kde )
        # END
        print( "Accuracy of the KDE classifier = %.1f%%" % test_accuracy_kde )

    print( "Best accuracy of the KDE classifier = %.1f%%" % best_kde_accuracy )

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
