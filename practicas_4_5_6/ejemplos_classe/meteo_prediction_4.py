
from __future__ import print_function

import os
import sys
import numpy
import pandas
import datetime

import machine_learning

import utils 

from matplotlib import pyplot

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, Activation
from keras.layers import Dropout, BatchNormalization, GaussianNoise
from keras.layers import Dense, LSTM
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD


"""
    0         1         2         3         4         5         6         7         8  
    012345678901234567890123456789012345678901234567890123456789012345678901234567890
                       |      |
      previous_history +      |
                       offset +
                              |
                      d_ahead +

        * target value at T[26] is going to be predicted with input values T[0:19+1],
          by leaving a number of days (offset = 7 in this example), optionally using 'd_ahead' days
          in order to give a context for the predictor.

    - 'previouos_history' is the number of days before and in addition the current one used to predict the target day.
    - 'offset' the number of days the target day is away the current one. Cannot be zero.
    - 'd_ahead' should be always lower than or equal to 'offset', and is the number of days to predict using 

    - 't0' the first day in T to be predicted or used for training, it will be 'previous_history+offset-1'
    - 't1' the day after the last day in T to be predicted. It will be allways equal to 'len(T)'

    * in the example, the target day T[26] = T[19+7] = T[previous_history+offset-1] is going to be
      predicted using the value in the days from 0 to 19, both included, that means 'previous_history'
      is equal to 20 in this case, and 'offset' equal to 7. So

        T[26] = T[ previous_history+offset-1 ] is predicted by using T[ 0 : 20 ] = T[ 0 : previous_history ]
    
      if 'd_ahead' is greater than 1 then

        T[ 26+1-d_ahead : 26+1 ] = T[ previous_history+offset-d_ahead : previous_history+offset ]
        is also predicted by using T[ 0 : 20 ] = T[ 0 : previous_history ]
"""

def show_errors( time, Y_true, Y_predict, with_graphs=False ):

    mae  = utils.mean_absolute_error(            Y_true, Y_predict )
    mape = utils.mean_absolute_percentage_error( Y_true, Y_predict, epsilon=1.0 )
    mse  = utils.mean_squared_error(             Y_true, Y_predict )

    print( 'MSE   %f ' % mse.mean() )
    print( 'MAE   %f ' % mae.mean() )
    print( 'MAPE  %7.3f%% ' % mape.mean() )

    if with_graphs:
        pyplot.plot( time, Y_predict[:,0], color='blue',  lw=7, alpha=0.2 )
        pyplot.plot( time, Y_predict[:,1], color='green', lw=7, alpha=0.2 )
        pyplot.plot( time, Y_predict[:,2], color='red',   lw=7, alpha=0.2 )
        pyplot.plot( time, Y_true[:,0], color='blue',  lw=2 )
        pyplot.plot( time, Y_true[:,1], color='green', lw=2 )
        pyplot.plot( time, Y_true[:,2], color='red',   lw=2 )
        pyplot.grid()
        pyplot.show()

        pyplot.plot( time, mape, color='red',   lw=1 )
        pyplot.grid()
        pyplot.show()


def prep_model( X, Y ):
    main_input = Input( shape=( X.shape[1], ), name='main_input' )
    layer = GaussianNoise(0.1)(main_input)
    layer = Dense( 512, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense( 512, activation='relu')(layer)
    main_output = Dense( Y.shape[1], activation='linear' )(layer)
    model = Model( inputs=[main_input], outputs=[main_output] )
    #
    optimizer = RMSprop()
    loss = keras.losses.mean_squared_error # for regression
    #loss = keras.losses.categorical_crossentropy # for classifying
    model.compile( loss=loss, optimizer=optimizer )
    model.summary()
    #
    return model


# MAIN

show_graphs=False

# Load data
df = pandas.read_csv( 'meteo_daily.csv', sep=';' )
fechas = utils.str_to_date( numpy.array( df['fecha'] ) )
T = numpy.zeros( [ len(fechas), 3 ] ) # Daily temperatures
T[:,0] = df['minima']
T[:,1] = df['media']
T[:,2] = df['maxima']


previous_days = 3
days_to_predict = 2
size_X = (len(T)-previous_days-days_to_predict+1)//1

X = numpy.zeros( [ size_X, T.shape[1]*previous_days ] )
Y = numpy.zeros( [ size_X, T.shape[1]*days_to_predict ] )

for t in range(len(X)):
    i=t*1
    X[t,:] = T[i              :i+previous_days                ,:].ravel()
    Y[t,:] = T[i+previous_days:i+previous_days+days_to_predict,:].ravel()


#pca=PCA(5)
#pca.fit( X )
#X = pca.transform(X)

pf = PolynomialFeatures(2)
pf.fit(X)
X=pf.transform(X)


N = int(0.8*len(X))
X_train = X[ :N]
Y_train = Y[ :N]
X_test  = X[N: ]
Y_test  = Y[N: ]

"""
knnr = KNeighborsRegressor( n_neighbors=7, weights='uniform', algorithm='kd_tree', leaf_size=30, p=2 )
knnr.fit( X_train, Y_train )
y_predict = knnr.predict( X_test )
"""
y_predict = numpy.zeros( [ len(X_test), Y_test.shape[1] ] )
regressors = []
for i in range(Y.shape[1]):
    brr = BayesianRidge( n_iter=300, fit_intercept=False, verbose=True )
    brr.fit( X_train, Y_train[:,i] )
    y_predict[:,i] = brr.predict( X_test )

show_errors( numpy.arange(len(Y_test)), Y_test[:  ,3:], y_predict[ :,3:], with_graphs=True )
show_errors( numpy.arange(len(Y_test)-2), Y_test[:-2,3:], y_predict[2:,3:], with_graphs=True )
#show_errors( fechas[N+previous_days:-1], Y_test[:  ,3:], y_predict[ :,3:], with_graphs=True )
#show_errors( fechas[N+previous_days:-3], Y_test[:-2,3:], y_predict[2:,3:], with_graphs=True )


if show_graphs:
    pyplot.plot( fechas[:], T[:,0], color='blue',  lw=1 )
    pyplot.plot( fechas[:], T[:,1], color='green', lw=1 )
    pyplot.plot( fechas[:], T[:,2], color='red',   lw=1 )
    pyplot.grid()
    pyplot.show()
