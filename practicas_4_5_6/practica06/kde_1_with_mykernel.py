
import numpy
from matplotlib import pyplot

from machine_learning import MyKernel
from machine_learning import MyKernelClassifier
from machine_learning import generate_datasets

if __name__ == "__main__" :

    X_train,Y_train,X_test,Y_test = generate_datasets.generate_multivariate_normals( 5, 2, 150, 150, 5.0, 2.0 )

    mykc = MyKernelClassifier()
    mykc.fit( X_train, Y_train )

    y_pred = mykc.predict( X_test )

    print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )
    print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )

    #fig,ax = pyplot.subplots( 1, 1, sharey=False )
    pyplot.scatter( X_train[:,0], X_train[:,1], c=Y_train, s=50, edgecolors='none' )
    pyplot.show()

