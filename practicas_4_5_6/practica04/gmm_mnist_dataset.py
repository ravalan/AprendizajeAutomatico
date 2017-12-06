
# Practica 4

import numpy
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

print( "Loading ... " )

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

X = mnist.data
Y = mnist.target

print( "Transforming  ... " )

new_X = X
#new_X = numpy.ndarray( [ len(X), 56 ] )
#for n in range(len(X)):
#    sample = X[n].reshape(28,28)
#    for i in range(28): new_X[n,i] = sample[i,:].sum()
#    for j in range(28): new_X[n,28+j] = sample[:,j].sum()
##    new_X[n] = new_X[n] / new_X[n].sum()

new_X = PCA(n_components=91).fit_transform(new_X)

#for n in range(len(new_X)): new_X[n] = new_X[n] / new_X[n].sum()
print( new_X[0] )

norm = StandardScaler()
#new_X = norm.fit_transform(new_X)


X_test = new_X[60000:]
Y_test = Y[60000:]

X_train = new_X[:60000]
Y_train = Y[:60000]

print( "Selecting  ... " )

num_classes = len(numpy.unique(mnist.target))
samples_per_class = []
for target in range(num_classes):
    samples_per_class.append( X_train[ Y_train==target ] )
    print( samples_per_class[-1].shape )

print( "Learning  phase 1 :: GMM per class ... " )

# Covariance valid types: 'diag', 'spherical', 'tied', 'full'

num_subclases=13
mixtures = []
for target in range(num_classes):
    print( "                     GMM for class %2d ... " % target )
    mixtures.append( GaussianMixture( n_components=num_subclases, covariance_type='full',
                            init_params='kmeans', max_iter=200, n_init=1 ) )
    mixtures[target].fit( samples_per_class[target] )

print( "Working with %d components per GMM" % num_subclases )

print( "Predicting  ... " )

densities = numpy.zeros( [ len(X_test), num_classes ] )

prioris = numpy.ones(num_classes)/num_classes # Wrong initialization

for target in range(num_classes):
    #prioris[target] = 1.0 * len(samples_per_class[target]) / len(X_train)
    #densities[:,target] = numpy.exp( mixtures[target].score( X_test ) )
    prioris[target] = numpy.log( len(samples_per_class[target]) ) - numpy.log( len(X_train) )
    densities[:,target] = mixtures[target].score_samples( X_test )

y_pred = numpy.zeros(len(X_test))
for n in range(len(X_test)):
    k = 0
    for target in range(num_classes):
        #if densities[n,target] * prioris[target] > densities[n,k] * prioris[k] :
        if densities[n,target] + prioris[target] > densities[n,k] + prioris[k] :
            k = target
    y_pred[n] = k


print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )

print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )
