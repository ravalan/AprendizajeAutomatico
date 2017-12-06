
import numpy
from matplotlib import pyplot

def gen_mean_and_sigma( n ):
    mean = 10*numpy.random.rand( 2 )
    sigma = numpy.zeros( [n,n] )
    for i in range(n):
        sigma[i,i] = (1+numpy.random.rand()*5)
        for j in range(i):
            sigma[i,j] = sigma[j,i] = numpy.random.rand()*2
    return mean, sigma

N=100

mean1, sigma1 = gen_mean_and_sigma(2)
mean2, sigma2 = gen_mean_and_sigma(2)
mean3, sigma3 = gen_mean_and_sigma(2)

x1 = numpy.random.multivariate_normal( mean1, sigma1, N )
x2 = numpy.random.multivariate_normal( mean2, sigma2, N )
x3 = numpy.random.multivariate_normal( mean3, sigma3, N )

pyplot.scatter( x1[:,0], x1[:,1], c='orange',  s=50, marker='o' )
pyplot.scatter( x2[:,0], x2[:,1], c='magenta', s=50, marker='D' )
pyplot.scatter( x3[:,0], x3[:,1], c='lime',    s=50, marker='s' )


pyplot.title( "Three Gaussian distributions in 2D" )
pyplot.xlabel( "1st dimension" )
pyplot.ylabel( "2nd dimension" )

pyplot.show()
