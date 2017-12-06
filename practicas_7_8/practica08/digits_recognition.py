"""
=====================================================
Digits recognition example using eigendigits and SVMs
=====================================================

The dataset used in this example is the MNIST
handwritten digit.

Expected results for the 10 digits training with the
whole training set.

SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0, degree=3,
  gamma=0.005, kernel='rbf', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)
    Predicting digits on the test set
    done in 75.882s

            precision    recall  f1-score   support

  Digit 0        0.98      0.99      0.99       980
  Digit 1        0.99      0.99      0.99      1135
  Digit 2        0.96      0.98      0.97      1032
  Digit 3        0.98      0.98      0.98      1010
  Digit 4        0.98      0.98      0.98       982
  Digit 5        0.98      0.98      0.98       892
  Digit 6        0.99      0.99      0.99       958
  Digit 7        0.98      0.97      0.98      1028
  Digit 8        0.98      0.98      0.98       974
  Digit 9        0.98      0.96      0.97      1009

avg / total       0.98      0.98      0.98     10000

[[ 974    0    2    1    0    0    1    0    2    0]
 [   0 1124    3    0    1    1    3    1    2    0]
 [   4    1 1010    2    2    0    2    4    5    2]
 [   0    0    7  986    0    5    1    5    5    1]
 [   0    0    7    0  964    0    1    0    0   10]
 [   2    0    2   10    1  870    3    0    4    0]
 [   3    2    0    0    1    4  948    0    0    0]
 [   0    2   11    2    2    1    0 1002    1    7]
 [   4    0    3    1    1    3    2    4  954    2]
 [   3    2    2    6   10    2    0   10    4  970]]
Accuracy 98.0%


"""
from __future__ import print_function

import sys
from time import time
import logging
import numpy
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

# introspect the images arrays to find the shapes (for plotting)
#n_samples, h, w = mnist.images.shape
n_samples,dim = mnist.data.shape
h = int(numpy.sqrt(dim))
w = h

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = mnist.data
n_features = X.shape[1]
X = X/255.0
#scaler = StandardScaler()
#X = scaler.fit_transform( X )

# the label to predict is the id of the person
y = mnist.target
#target_digits = lfw_people.target_digits
target_digits = numpy.array( [ ' Digit {:.0f} '.format( value ) for value in numpy.unique(y) ] )
n_classes = target_digits.shape[0]

y = y.astype(numpy.int64)

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
X_train = X[:60000 ]
y_train = y[:60000 ]
X_test  = X[ 60000:]
y_test  = y[ 60000:]

(X_train,y_train) = shuffle( X_train, y_train )
X_train = X_train[:60000 ]
y_train = y_train[:60000 ]

(X_test,y_test) = shuffle( X_test, y_test )


###############################################################################
# Compute a PCA (eigendigits) on the digit dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 30

print("Extracting the top %d eigendigits from %d digits" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigendigits = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigendigits orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

scaler = StandardScaler()
X_train_pca = scaler.fit_transform( X_train_pca )
X_test_pca = scaler.transform( X_test_pca )


###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1, 10, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
param_grid = {'C': [10], 'gamma': [0.005], }
clf = GridSearchCV( SVC(kernel='rbf', class_weight='balanced', max_iter=1000), param_grid )
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting digits on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_digits))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

accuracy = ( 100.0 * (y_test == y_pred).sum() ) / len(y_test)
print( "Accuracy %.1f%%" % accuracy )


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery( images, titles, h, w, n_row=3, n_col=4 ):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_digits, i):
    pred_digit = target_digits[y_pred[i]] # .rsplit(' ', 1)[-1]
    true_digit = target_digits[y_test[i]] # .rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_digit, true_digit)

prediction_titles = [ title( y_pred, y_test, target_digits, i ) for i in range(y_pred.shape[0]) ]

plot_gallery( X_test, prediction_titles, h, w )

# plot the gallery of the most significative eigendigits

eigendigit_titles = ["eigendigit %d" % i for i in range( eigendigits.shape[0] ) ]
plot_gallery( eigendigits, eigendigit_titles, h, w )

plt.show()
