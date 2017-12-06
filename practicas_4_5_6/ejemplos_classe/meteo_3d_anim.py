"""
============
3D animation
============

A simple example of an animated plot... In 3D!
"""
import numpy
import pandas
#
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib

from utils import str_to_date
from machine_learning import KMeans


def load_meteo_data( filename ):
    df = pandas.read_csv( filename, sep=';' )
    dates = str_to_date( numpy.array( df['fecha'] ) )
    T = numpy.zeros( [ len(dates), 3 ] ) # Daily temperatures
    T[:,0] = df['minima']
    T[:,1] = df['media']
    T[:,2] = df['maxima']
    #
    return dates, T
    #


def update_colors( iteration, data, colors, scatter ):
    if iteration == 0 : kmeans.fit(T)
    changes=kmeans.fit_iteration(T)
    y=kmeans.predict(T)
    for n in range(len(colors)):
        """
        i=numpy.random.randint( len(colors) )
        #j=numpy.random.randint( len(colors) )
        k=numpy.random.randint( len(palette) )
        #temp = colors[i,:]
        #colors[i,:] = colors[j,:]
        #colors[j,:] = temp
        colors[i,:] = palette[k,:]
        """
        colors[n,:] = palette[y[n],:]
    print( "iteration = %3d clusters = %3d changes = %12d   J = %20.8f  %.8f" % ( iteration, kmeans.n_clusters, changes, kmeans.J, kmeans.improvement() ) )
    return scatter

#
#
#
#cmap = matplotlib.colors.Colormap( 'jet', N=10 )

dates,T = load_meteo_data( 'meteo_daily.csv' )

#T[:,:]=numpy.random.randn( len(T), 3 )*5 + 5

kmeans = KMeans( n_clusters=5+numpy.random.randint(20), init='random', number_of_initializations=1, max_iter=1 )
kmeans.fit( T )

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D( fig )

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
K=kmeans.n_clusters
color_index=numpy.random.randint( K, size=len(T) )
color_index[0:K] = numpy.arange(K)
scatter = ax.scatter( T[:,0], T[:,1], T[:,1], zdir='z', s=30, c=color_index, depthshade=True, linewidths=0 )
colors = scatter.get_facecolors()
palette = colors[:K].copy()
#print(palette)

# Setting the axes properties
ax.set_xlim3d( [-5.0, 20.0] )
ax.set_xlabel( 'X' )

ax.set_ylim3d( [-1.0, 40.0] )
ax.set_ylabel( 'Y' )

ax.set_zlim3d( [-1.0, 40.0] )
ax.set_zlabel( 'Z' )

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation( fig, update_colors, 50, fargs=(T, colors, scatter), interval=50, blit=False )

plt.show()
