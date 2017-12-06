"""
"""
from __future__ import print_function

import numpy
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop



print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print( x_train.shape[0], 'train samples', x_train.shape )
print( x_test.shape[0], 'test samples', x_test.shape )

# convert class vectors to binary class matrices

# positions info is ignored by this model)
X_train = x_train.reshape( -1, 28, 28, 1 )
X_test = x_test.reshape( -1, 28, 28, 1 )

# the label to predict is the id of the person
target_names = numpy.array( [ ("Digit %d" % i) for i in numpy.unique(y_test) ] )
print( target_names )
n_classes = target_names.shape[0]
n_features = x_train.shape[1] * x_train.shape[2]
h = x_train.shape[1]
w = x_train.shape[2]

print("Total dataset size:")
print("n_samples: %d  of %d x %d" % (len(x_train), h, w ) )
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

print( 'X.shape', X_train.shape )
print( 'X.min()', X_train.min() )
print( 'X.max()', X_train.max() )


###############################################################################
# Train a ANN classification model

# Preparing one-hot vector
Y_train = numpy.zeros( [ len(y_train), n_classes ] )
Y_test  = numpy.zeros( [ len(y_test),  n_classes ] )

Y_train[ numpy.arange(len(y_train)), y_train ] = 1
Y_test[  numpy.arange(len(y_test)),  y_test  ] = 1

add_noise=True

model = Sequential()
if add_noise:
    model.add( GaussianNoise( 0.3, input_shape=(X_train.shape[1],X_train.shape[2],1) ) )
    model.add( Conv2D( filters=17, kernel_size=(7,7), strides=(1,1), activation='relu') )
    #model.add( Conv2D( filters=2, kernel_size=(3,3), activation='relu') )
    #model.add( Conv2D( filters=7, kernel_size=(4,4), strides=(2,2), activation='relu' ) )
else:
    model.add( Conv2D( filters=7, kernel_size=(4,4), strides=(2,2), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)) )
    #model.add( Conv2D( filters=2, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)) )
    #model.add( Conv2D( filters=3, kernel_size=(7,7), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)) )
model.add( Flatten() )
model.add( Dropout(0.5) )
#model.add( BatchNormalization() )
model.add( Dense( 4024, activation='relu') )
model.add( Dropout(0.5) )
#model.add( BatchNormalization() )
model.add( Dense( 1024, activation='relu') )
model.add( Dropout(0.5) )
#model.add( BatchNormalization() )
model.add( Dense( n_classes, activation='softmax') )
model.summary()
model.compile( loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'] )


print("Fitting the classifier to the training set")
batch_size=100
epochs=5
t0=time()
history = model.fit( X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test) )
print("done in %0.3fs" % (time() - t0))
score = model.evaluate( X_test, Y_test, verbose=0 )
print('Test loss:', score[0])


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = model.predict( X_test )
y_pred = y_pred.argmax(axis=1)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery( images, titles, h, w, n_row=3, n_col=4 ):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        #plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

plot_data=False
if plot_data : 
    plot_gallery(X_test, prediction_titles, h, w)

    # plot the gallery of the most significative eigenfaces

    if eigenfaces is not None:
        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()
