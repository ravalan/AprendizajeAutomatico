"""
================
The Bayes update
================

This animation displays the posterior estimate updates as it is refitted when
new data arrives.
The vertical line represents the theoretical value to which the plotted
distribution should converge.
"""

# update a distribution based on new data.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import scipy.stats as ss
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import KernelDensity


class UpdateDist(object):
    def __init__( self, ax, N=200 ):
        self.N = N
        #np.random.seed(1)
        self.X = np.concatenate((np.random.normal(0, 1, int(0.3 * self.N)),
                                 np.random.normal(5, 1, int(0.7 * self.N))))[:, np.newaxis]

        self.X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

        self.true_dens = (0.3 * ss.norm(0, 1).pdf(self.X_plot[:, 0])
                        + 0.7 * ss.norm(5, 1).pdf(self.X_plot[:, 0]))

        self.ax = ax

        self.ax.fill( self.X_plot[:, 0], self.true_dens, fc='black', alpha=0.2, label='input distribution' )

        self.X = np.concatenate( (np.random.normal(0, 1, int(0.3 * self.N)),
                                  np.random.normal(5, 1, int(0.7 * self.N))) )[:, np.newaxis]
        self.X = shuffle(self.X)
        self.Y = -0.01 - 0.02 * np.random.rand(len(self.X))
        #
        self.ax.text( 6, 0.38, "N={0} points".format(self.N) )

        self.colors=dict( gaussian='blue', tophat='green', epanechnikov='red' )
        self.lines=dict()

        for kernel in ['gaussian', 'tophat', 'epanechnikov']:
            kde = KernelDensity( kernel=kernel, bandwidth=0.5 ).fit(self.X[:1])
            log_dens = kde.score_samples(self.X_plot)
            self.lines[kernel], = self.ax.plot( self.X_plot[:, 0], np.exp(log_dens), '-', color=self.colors[kernel], label="kernel = '{0}'".format(kernel) )
        self.lines['samples'], = self.ax.plot( self.X[:1,0], self.Y[:1], '+k' )

        self.ax.set_xlim( -4, 9 )
        self.ax.set_ylim( -0.04, 0.4 )
        self.ax.legend( loc='upper left' )
        self.ax.grid(True)

    def init(self):
        for kernel in ['gaussian', 'tophat', 'epanechnikov']:
            kde = KernelDensity( kernel=kernel, bandwidth=0.5 ).fit(self.X[:1])
            log_dens = kde.score_samples(self.X_plot)
            self.lines[kernel].set_data( self.X_plot[:,0], np.exp(log_dens) )
        self.lines['samples'].set_data( self.X[:1,0], self.Y[:1] )

        return self.lines['gaussian'], self.lines['tophat'], self.lines['epanechnikov'], self.lines['samples'],

    def __call__( self, i ):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            return self.init()

        print( "%10d / %d" % (i,self.N), end='\r' )
        for kernel in ['gaussian', 'tophat', 'epanechnikov']:
            kde = KernelDensity( kernel=kernel, bandwidth=0.5 ).fit(self.X[:i+1])
            log_dens = kde.score_samples(self.X_plot)
            self.lines[kernel].set_data( self.X_plot[:,0], np.exp(log_dens) )
        self.lines['samples'].set_data( self.X[:i+1,0], self.Y[:i+1] )

        return self.lines['gaussian'], self.lines['tophat'], self.lines['epanechnikov'], self.lines['samples'],


fig, ax = plt.subplots( figsize=(10,8) )
ud = UpdateDist( ax, N=500 )
anim = FuncAnimation( fig, ud, frames=np.arange(ud.N), init_func=ud.init, interval=max(10,10000//ud.N), blit=True )
plt.show()
