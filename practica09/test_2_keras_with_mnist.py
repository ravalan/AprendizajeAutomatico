'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization, Input
from keras.optimizers import RMSprop
from keras.constraints import maxnorm


batch_size = 128
num_classes = 10
epochs = 5

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical( y_train, num_classes )
y_test = keras.utils.to_categorical( y_test, num_classes )
"""
_y_train = numpy.zeros( [ len(y_train), num_classes ], dtype=bool )
_y_test  = numpy.zeros( [ len(y_test),  num_classes ], dtype=bool )
for i in range(len(y_train)): _y_train[ y_train[i] ] = True
for i in range(len(y_test )): _y_test[   y_test[i] ] = True
y_train = _y_train
y_test  = _y_test
"""

model = Sequential()
#model.add( Input( shape=(784,) ) )
#model.add( Dense( 784, activation='linear', input_shape=(784,)) )
#model.add( GaussianNoise( 0.3, input_shape=(784,) ) )
#model.add( Dense( 512, activation='relu', input_shape=(784,)) )
model.add( Dense( 512, activation='relu', kernel_constraint=maxnorm(3.0), input_shape=(784,)) )
model.add( Dropout(0.3) )
#model.add( BatchNormalization() )
model.add( Dense( 256, activation='relu', kernel_constraint=maxnorm(3.0) ) )
model.add( Dropout(0.3) )
#model.add( BatchNormalization() )
model.add( Dense( 256, activation='relu', kernel_constraint=maxnorm(3.0) ) )
model.add( Dropout(0.3) )
#model.add( BatchNormalization() )
model.add( Dense( 256, activation='relu', kernel_constraint=maxnorm(3.0) ) )
model.add( Dropout(0.3) )
#model.add( BatchNormalization() )
model.add( Dense( 256, activation='relu', kernel_constraint=maxnorm(3.0) ) )
model.add( Dropout(0.3) )
#model.add( BatchNormalization() )
model.add( Dense( 10, activation='softmax', kernel_constraint=maxnorm(3.0) ) )

model.summary()

model.compile( loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'] )

history = model.fit( x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_data=(x_test, y_test) )
#history = model.fit( x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(x_test, y_test) )
score = model.evaluate( x_test, y_test, verbose=0 )
print('Test loss:', score[0])
print('Test accuracy:', score[1])

