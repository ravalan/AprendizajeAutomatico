
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
    mape = utils.mean_absolute_percentage_error( Y_true, Y_predict )
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


previous_days = 6
X = numpy.zeros( [ len(T)-previous_days, T.shape[1]*previous_days ] )
Y = numpy.zeros( [ len(T)-previous_days, T.shape[1] ] )
for t in range(len(X)):
    X[t,:] = T[t:t+previous_days,:].ravel()
    Y[t,:] = T[  t+previous_days,:]
    #print( "t=%d   t+previous_days=%d   t+previous_days+1=%d" % ( t, t+previous_days, t+previous_days+1 ) )
    #print( "X[t] ", X[t] )
    #print( "Y[t] ", Y[t] )


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


knnr = KNeighborsRegressor( n_neighbors=3, weights='uniform', algorithm='kd_tree', leaf_size=30, p=2 )

knnr.fit( X_train, Y_train )
y_predict = knnr.predict( X_test )

y_predict_2 = numpy.zeros( y_predict.shape )

dim=T.shape[1]
for t in range(1,len(X_test)):
    x = T[N+t-1:N+t-1+previous_days,:]
    #print( x.shape )
    x = x.ravel().reshape(1,-1)
    #print( x.shape )
    x = pf.transform( x )
    #print( x.shape )
    y = knnr.predict( x )
    x = T[N+t:N+t+previous_days,:].copy()
    x[-1,:] = y[:]
    x = x.ravel().reshape(1,-1)
    x = pf.transform( x )
    y_predict_2[t] = knnr.predict( x )



show_errors( fechas[N+previous_days:], Y_test, y_predict, with_graphs=True )
show_errors( fechas[N+previous_days+1:], Y_test[1:], y_predict_2[1:], with_graphs=True )
show_errors( fechas[N+previous_days:-3], Y_test[:-3], y_predict_2[3:], with_graphs=True )


if show_graphs:
    pyplot.plot( fechas[:], T[:,0], color='blue',  lw=1 )
    pyplot.plot( fechas[:], T[:,1], color='green', lw=1 )
    pyplot.plot( fechas[:], T[:,2], color='red',   lw=1 )
    pyplot.grid()
    pyplot.show()
