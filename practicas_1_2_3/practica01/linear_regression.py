
import sys
import numpy
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import pipeline
from matplotlib import pyplot

# --------------------------------------------------
def f(x):
    return numpy.sin( 2 * numpy.pi * x )
# --------------------------------------------------

# --------------------------------------------------
def g(x):
    return f(x) + 0.25*numpy.random.randn( len(x) )
# --------------------------------------------------

# --------------------------------------------------
def my_error(real,estimation):
    temp_ = real - estimation
    return numpy.sqrt( 2 * numpy.dot(temp_,temp_) / len(temp_) ) # This method computes the Root Mean Square (RMS) error
    # return 0.5 * numpy.dot(temp_,temp_) / len(temp_) # This computes the Mean Square Error (MSE)
# --------------------------------------------------


# --------------------------------------------------
# MAIN
# --------------------------------------------------

num_samples=8
max_degree=12
verbose=True
intermediate_graphics=True

# Generation of the true function for representing it.
zx = numpy.linspace(0.0,1.0,1000)
zy = f(zx)
zx_mat = zx.reshape( len(zx), 1 ) # Converts zx to a matrix with one column. It is needed later in some functions.

# Computation of the limits on y-axis for representing the data.
min_y=zy.min()
max_y=zy.max()
dy = max_y-min_y
min_y = min_y - 0.1 * dy
max_y = max_y + 0.1 * dy

# Generation of the training-set and the test-set
#X_train = numpy.linspace( 0.05, 0.95, num_samples ) # Values for x-axis equally spaced
X_train = numpy.random.rand(num_samples) # Purely random values for x-axis, choose this for observing the behaviour of the technique with few samples in the training set
X_test = numpy.random.rand( min( 1000, num_samples*10 ) ) # Enough values for the x-axis for the test-set

# Computation the funcion with noise. See the definition of g() above.
Y_train = g(X_train)
Y_test  = g(X_test)
# Computation of the true function. See the definition of f() above.
Y_train_true = f(X_train)
Y_test_true  = f(X_test)


# This is necessary for the fit method, then, accessing one column of matrix X_train_mat is X_train_mat[:,column]
# X_train is also used for representing the scatter-plot.
X_train_mat=X_train.reshape( len(X_train), 1 )
X_test_mat = X_test.reshape( len(X_test),  1 )

# Preparation of the array for maintaining the evolution of the error.
error_polynomial = numpy.ones( [max_degree+1,2] )

for degree in range(max_degree+1):
    #
    # Creation of the object for transforming the input variables to a polynomial combination of them.
    #
    poly = PolynomialFeatures( degree = degree )
    #
    # PolynomialFeatures.fit_transform() needs a bidimensional array or matrix, with as many rows as samples and as many columns as the dimensionality of samples.
    # That's why we do the reshape() some lines above.
    #
    X_  = poly.fit_transform( X_train_mat )
    zx_ = poly.fit_transform( zx_mat )
    #
    # Creation of an object of the class LinearRegresion.
    # Choose one of the following according to the run you are testing.
    # You can test different values of alpha in the Ridge Linear Regression model.
    #
    linear_regressor = linear_model.LinearRegression()
    #linear_regressor = linear_model.Ridge(alpha=0.001)
    #
    # The fit() method trains the model.
    # The transformed input samples are used in order to achieve
    # a polynomial regressor by using a linear one.
    #
    linear_regressor.fit( X_, Y_train )

    if verbose:
        print( 'Polynomial regression of degree %d: ' % degree )
        print( 'Intercept: ' + str(linear_regressor.intercept_) )
        print( 'Coef.....: ' + str(linear_regressor.coef_ ) )

    if intermediate_graphics:
        #pyplot.figure(degree+1)
        pyplot.title( "Polynomial regression example 1 with degree = %d" % degree )
        pyplot.xlabel( "x" )
        pyplot.ylabel( "sin(2*PI*x)" )
        plot1 = pyplot.plot( zx, zy, c='g' )
        plot2 = pyplot.plot( X_train, Y_train, 'ro' )
        plot3 = pyplot.plot( zx, linear_regressor.predict(zx_), c='m' )
        pyplot.ylim( min_y, max_y )
        pyplot.show()

#if intermediate_graphics:
#    pyplot.show()

    #
    # For each model (degree of the polynomial regressor based on an implementation of a linear regressor)
    # we record the error of the prediction with respect to the true function, both for train and test samples.
    #
    error_polynomial[degree][0] = my_error( Y_train_true, linear_regressor.predict( X_ ) )
    error_polynomial[degree][1] = my_error( Y_test_true,  linear_regressor.predict( poly.fit_transform(X_test_mat) ) )

if verbose:
    for degree in range(len(error_polynomial)):
        print( "Error polynomial (%d) = %17.10f %17.10f" % (degree, error_polynomial[degree][0], error_polynomial[degree][1] ) )


#
# Finally, we represent the evolution of the error as the degree of the polynomial transformation of input samples grows.
#
pyplot.title( "Evolution of Root Mean Square (RMS) error" )
pyplot.xlabel( "Degree" )
pyplot.ylabel( "RMS" )
pyplot.grid()
#
artists=[]
labels=[]
#
plot1 = pyplot.plot( numpy.arange(len(error_polynomial)), error_polynomial[:,0], c='b' )
artists.append( plot1[0] )
labels.append( "Error on training set " )
#
plot2 = pyplot.plot( numpy.arange(len(error_polynomial)), error_polynomial[:,1], c='r' )
artists.append( plot2[0] )
labels.append( "Error on test set " )
#
dy = error_polynomial[:,1].max()
pyplot.ylim( -0.1*dy, 1.1*dy )
#
pyplot.legend( artists, labels )
#
pyplot.show()
