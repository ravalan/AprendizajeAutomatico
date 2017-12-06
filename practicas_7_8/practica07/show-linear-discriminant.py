
import numpy
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot

from machine_learning import MyKernelClassifier
from machine_learning import LinearDiscriminant
from machine_learning import generate_datasets

# --------------------------------------------------------------------
def compute_hyperplane( X, Y ):
    m1 = numpy.mean( X[ Y==1 ], axis=0 )
    m2 = numpy.mean( X[ Y==0 ], axis=0 )

    # General equation
    A = - ( m1[1] - m2[1] )
    B =   ( m1[0] - m2[0] )
    C = - A * m1[0] - B * m1[1]

    # Intermediate point
    ip = (m1+m2)/2.0

    return A, B, C, ip, m1, m2
# --------------------------------------------------------------------

# --------------------------------------------------------------------
def compute_hyperplane_fisher( X, Y ):
    
    # The mean vector of each class
    m1 = numpy.mean( X[ Y==1 ], axis=0 )
    m2 = numpy.mean( X[ Y==0 ], axis=0 )

    # Sw stores the variance whithin classes, it is accumulated for all classes
    Sw = numpy.zeros( X.shape[1] )
    for n in range(len(X)):
        if Y[n] == 1 :
            Sw = Sw + numpy.outer( (X[n] - m1 ), (X[n] - m1 ) )
        else: 
            Sw = Sw + numpy.outer( (X[n] - m2 ), (X[n] - m2 ) )

    invSw = numpy.linalg.inv(Sw)
    w = numpy.dot( invSw, (m2-m1) )
    w = w / numpy.sqrt(numpy.dot(w,w))

    # General equation for 'w' as orthogonal to the hyperplane
    A = -w[1]
    B =  w[0]
    C = - A * m1[0] - B * m1[1]

    # Intermediate point
    ip = (m1+m2)/2.0

    return A, B, C, ip, m1, m2
# --------------------------------------------------------------------


if __name__ == "__main__" :

    X_train,Y_train,X_test,Y_test = generate_datasets.generate_multivariate_normals( 2, 2, 150, 50, 5.0, 2.0 )

    pyplot.scatter( X_train[Y_train==1,0], X_train[Y_train==1,1], c='orange', s=50, edgecolors='none' )
    pyplot.scatter( X_train[Y_train==0,0], X_train[Y_train==0,1], c='cyan', s=50, edgecolors='none' )

    # fix the limits in order to show the data in isotropic axes
    xmin,xmax = pyplot.xlim()
    ymin,ymax = pyplot.ylim()
    xmin=ymin=min(xmin,ymin)
    xmax=ymax=max(xmax,ymax)
    pyplot.xlim( xmin, xmax )
    pyplot.ylim( ymin, ymax )

    colors=['red','green','blue','magenta','yellow']
    zx = numpy.linspace(xmin,xmax,1000)

    # SIMPLE HIPERPLANE
    A,B,C,ip,m1,m2 = compute_hyperplane( X_train, Y_train )
    # slope and intercept of the hyperplane that is orthogonal to 'w'
    slope =  B/A
    intercept = ip[1] - slope*ip[0]
    # draw the hyperplane
    pyplot.plot( zx, slope*zx+intercept, color=colors[0], lw=3, ls='--' );

    # FISHER'S HIPERPLANE
    A,B,C,ip,m1,m2 = compute_hyperplane_fisher( X_train, Y_train )
    # slope and intercept of the hyperplane that is orthogonal to 'w'
    slope =  B/A
    intercept = ip[1] - slope*ip[0]
    # draw the hyperplane
    pyplot.plot( zx, slope*zx+intercept, color=colors[2], lw=3, ls='--' );

    # draw the line from mean 1 to mean 2
    x0 = m1[0] ; y0 = m1[1]
    x1 = m2[0] ; y1 = m2[1]
    pyplot.plot( [x0,x1], [y0,y1], color=colors[3], lw=2 );

    # compute the unitary 'w' given the general equation of it as orthogonal to the hyperplane
    w=numpy.zeros( 2 )
    w[0] = -B
    w[1] =  A
    w = w / numpy.sqrt(numpy.dot(w,w))
    
    # decides were to show 'w'
    x0 = 0.75*(xmax+xmin) ; y0 = slope * x0 + intercept
    if y0 > ymax or y0 < ymin :
        y0 = 0.75*(xmax+xmin)
        x0 = (y0 - intercept)/slope
        w = 0.25*(ymax-ymin)*w
    else:
        w = 0.25*(xmax-xmin)*w

    # draw the arrow for 'w'
    pyplot.annotate( 'w', xy=(x0,y0), xycoords='data', xytext=(x0+w[0],y0+w[1]), textcoords='data', arrowprops=dict(arrowstyle='<-') )

    pyplot.grid()
    pyplot.show()
