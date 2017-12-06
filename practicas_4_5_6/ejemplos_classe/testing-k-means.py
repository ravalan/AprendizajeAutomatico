
import time
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation

#from sklearn.cluster import KMeans
from machine_learning import KMeans


class AnimatedClustering(object):

    def __init__( self, X ):
        self.num_clusters=10
        self.X_ = X
        self.data_ = numpy.zeros( (2, X.shape[0]+self.num_clusters) )
        self.data_[0,:len(X)] = X[:,0]
        self.data_[1,:len(X)] = X[:,1]
        #self.kmeans = KMeans( self.num_clusters, init='Katsavounidis', n_init=1, max_iter=1 )
        self.kmeans = KMeans( n_clusters=self.num_clusters, init='random', number_of_initializations=1, max_iter=1 )
        self.kmeans.fit( self.X_ )
        self.Y_ = numpy.ones( len(X), dtype='int' )

        self.fig_, self.ax_ = pyplot.subplots()
        self.ani_ = animation.FuncAnimation( self.fig_, self.update_figure, self.generate_data, init_func=self.setup_plot, interval=1000, blit=True, repeat=False )
        self.changes_=0

    def setup_plot( self ):
        self.colour_ = numpy.ones( len(self.X_)+self.num_clusters )*3
        self.sizes_ = numpy.ones( self.X_.shape[0]+self.num_clusters ) * 30
        self.colour_[len(self.X_):] = self.num_clusters+1
        self.sizes_[len(self.X_):] = 100
        self.scat_ = self.ax_.scatter( self.data_[0,:], self.data_[1,:], c=self.colour_, s=self.sizes_, marker='o', edgecolors='none', animated=False )
        return self.scat_,

    def generate_data( self ):
        self.changes_ = len(self.X_)
        #self.kmeans.lloyd( self.X_, num_iter=1 )
        self.kmeans.fit_iteration( self.X_ )
        self.colour_[:len(self.X_)] = self.kmeans.predict( self.X_ )
        self.data_[0,len(self.X_):] = self.kmeans.cluster_centers_[:,0]
        self.data_[1,len(self.X_):] = self.kmeans.cluster_centers_[:,1]
        yield self.data_, self.colour_, self.sizes_

        while self.changes_ > 0 :
            #self.changes_ = self.kmeans.lloyd( self.X_, num_iter=1 )
            self.changes_ = self.kmeans.fit_iteration( self.X_ )
            self.Y_[:] = self.kmeans.predict( self.X_ )
            self.colour_[:len(self.Y_)] = self.Y_[:]
            self.data_[0,len(self.X_):] = self.kmeans.cluster_centers_[:,0]
            self.data_[1,len(self.X_):] = self.kmeans.cluster_centers_[:,1]
            yield self.data_, self.colour_, self.sizes_

    def update_figure( self, generated_data ):
        data,colour,sizes = generated_data
        print( "clusters = %d changes = %12d   J = %20.8f  %.8f" % ( self.num_clusters, self.changes_, self.kmeans.J, self.kmeans.improvement() ) )

        pyplot.clf()
        #pyplot.set_axis_bgcolor( 'white' )
        self.scat_ = self.ax_.scatter( self.data_[0,:], self.data_[1,:], c=colour, s=sizes, marker='o', edgecolors='none', animated=False )
        #pyplot.draw()

        return self.scat_,

    def show( self ):
        pyplot.show()


## MAIN

if __name__ == '__main__' :

    N=200
    x1 = numpy.random.rand( N, 2 )*100-50
    x2 = numpy.random.rand( N, 2 )*10-5
    #x1 = numpy.array( [ [0.0,1.0] ] )
    #x2 = numpy.array( [ [1.0,0.0] ] )
    x3 = numpy.random.multivariate_normal( [20.0,-46.0], [ [3.0, -2.0], [-2.0,3.0] ],  N )
    x4 = numpy.random.multivariate_normal( [-30.0,-50.0], [ [5.0, 0.0], [0.0,5.0] ],  N )
    x5 = numpy.random.multivariate_normal( [65.0,0.0], [ [1.0, 0.0], [0.0,1.0] ],  10 )

    fig,ax = pyplot.subplots( 1, 1, sharey=False )
    pyplot.scatter( x1[:,0], x1[:,1], c='r', s=50, edgecolors='none' )
    pyplot.scatter( x2[:,0], x2[:,1], c='g', s=50, edgecolors='none' )
    pyplot.scatter( x3[:,0], x3[:,1], c='b', s=50, edgecolors='none' )
    pyplot.scatter( x4[:,0], x4[:,1], c='m', s=50, edgecolors='none' )
    pyplot.scatter( x5[:,0], x5[:,1], c='y', s=50, edgecolors='none' )
    pyplot.show()

    X = numpy.vstack( [x1, x2, x3, x4, x5] )

    ac = AnimatedClustering(X)
    ac.show()

    print( ac.kmeans.cluster_centers_ )


#mydad = dad.DensitiesAndDistances( 5, 'euclidean' )
#mydad.find_clusters( X )
