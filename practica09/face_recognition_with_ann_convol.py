"""
===================================================
Faces recognition example using eigenfaces and ANNs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset::

                     precision    recall  f1-score   support

  Gerhard_Schroeder       0.91      0.75      0.82        28
    Donald_Rumsfeld       0.84      0.82      0.83        33
         Tony_Blair       0.65      0.82      0.73        34
       Colin_Powell       0.78      0.88      0.83        58
      George_W_Bush       0.93      0.86      0.90       129

        avg / total       0.86      0.84      0.85       282


 --- INITIAL COMMENT ---

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

lfw_people = fetch_lfw_people( min_faces_per_person=70, resize=0.4 )

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
#X = lfw_people.data
X = lfw_people.images
X = X.reshape( -1, h, w, 1 )
n_features = h*w

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
print( target_names )
print( target_names.shape )
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d  of %d x %d" % (n_samples, h, w ) )
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

X = X/255.0

print( 'X.shape', X.shape )
print( 'X.min()', X.min() )
print( 'X.max()', X.max() )


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)


###############################################################################
# Train a ANN classification model

# Preparing one-hot vector
Y_train = numpy.zeros( [ len(y_train), n_classes ] )
Y_test  = numpy.zeros( [ len(y_test),  n_classes ] )

Y_train[ numpy.arange(len(y_train)), y_train ] = 1
Y_test[  numpy.arange(len(y_test)),  y_test  ] = 1

add_noise=False

model = Sequential()
if add_noise:
    model.add( GaussianNoise( 0.3, input_shape=(X_train.shape[1],X_train.shape[2],1) ) )
    #model.add( Conv2D( filters=7, kernel_size=(5,5), activation='relu') )
    model.add( Conv2D( filters=2, kernel_size=(3,3), activation='relu') )
else:
    #model.add( Conv2D( filters=7, kernel_size=(5,5), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)) )
    model.add( Conv2D( filters=7, kernel_size=(5,5), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)) )
model.add( Flatten() )
#model.add( Dropout(0.5) )
#model.add( BatchNormalization() )
model.add( Dense( 512, activation='relu') )
#model.add( Dropout(0.5) )
#model.add( BatchNormalization() )
model.add( Dense( n_classes, activation='softmax') )
model.compile( loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'] )
model.summary()


print("Fitting the classifier to the training set")
batch_size=100
epochs=30
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
