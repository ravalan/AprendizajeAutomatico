
import sys
import numpy
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
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
#norm = StandardScaler(); X = norm.fit_transform(X)

X=X/255.0


# Separate test set and training set
X_test = X[60000:]
Y_test = Y[60000:]

X_train = X[:60000]
Y_train = Y[:60000]

(X_train,Y_train) = shuffle( X_train, Y_train )

pca = PCA(n_components=37)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# Uncomment these two lines for working with a subset
#X_train = X_train[:10000]
#Y_train = Y_train[:10000]

num_classes = len(numpy.unique(mnist.target))
samples_per_class = []
for target in range(num_classes):
    samples_per_class.append( X_train[ Y_train==target ] )
    print( "Class %3d with %10d " % ( (target+1), len(samples_per_class[target]) ) )

kernel='gaussian'
estimators=[]

h=X_train.shape[1] / 2 # choose the value of the 'h' parameter properly.
h=1.0

for k in range(num_classes):
    estimators.append( KernelDensity( kernel=kernel, bandwidth=h ).fit(samples_per_class[k]) )

y_pred = numpy.zeros( len(X_test) )
best_log_dens = numpy.zeros( len(X_test) )
for k in range(num_classes):
    print( "Scoring test samples for class %3d " % k )
    log_dens = estimators[k].score_samples(X_test)
    if 0 == k :
        best_log_dens[:] = log_dens[:]
        y_pred[:] = 0
    else:
        print( "Classifying test samples up to class %3d " % k )
        for n in range(len(X_test)):
            print( "\r %10d " % n, end='' )
            if log_dens[n] > best_log_dens[n]:
                best_log_dens[n] = log_dens[n]
                y_pred[n] = k
        print( " " )
    
print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )


