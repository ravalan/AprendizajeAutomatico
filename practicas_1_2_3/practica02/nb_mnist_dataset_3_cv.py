
import numpy
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import GaussianNB

print( "Loading ... " )

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )


X = mnist.data
Y = mnist.target

print( "Transforming  ... " )

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

num_cv = 10
size_cv = len(X_train)//num_cv
best_gnb = None
best_accuracy = 0.0
avg_accuracy = 0.0

for trial in range(num_cv):
    print( "\t\tCross validation trial %02d ... " % (trial+1) )

    from_sample = trial * size_cv
    to_sample = from_sample + size_cv

    print( from_sample, to_sample )

    # Extract the validation set
    X_validation = X_train[from_sample:to_sample]
    Y_validation = Y_train[from_sample:to_sample]

    # Extract the training set by means of a mask for do not use the validation set
    mask = numpy.ones( len(X_train), dtype=bool )
    mask[from_sample:to_sample] = False
    X_train_cv = X_train[ mask ]
    Y_train_cv = Y_train[ mask ]

    print( X_train_cv.shape )
    print( X_validation.shape )

    gnb = GaussianNB()
    gnb.fit( X_train_cv, Y_train_cv )
    y_pred = gnb.predict( X_validation )
    accuracy = ( 100.0 * (Y_validation == y_pred).sum() ) / len(Y_validation)
    print( "\t\tAccuracy = %.1f%%" % accuracy )

    avg_accuracy += accuracy

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_gnb = gnb

avg_accuracy /= num_cv
print( "\nAccuracy in average = %.1f%%\n" % avg_accuracy )


print( "Predicting  ... " )

y_pred = best_gnb.predict( X_test )

print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )

print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )
