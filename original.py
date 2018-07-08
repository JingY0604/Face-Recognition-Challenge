from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt


df = fetch_lfw_people(min_faces_per_person=20, resize=0.6)
# Resizing the image to 0.6 times its original and with a condition of min no. of pics available per person to 50

no_img, height, width = df.images.shape
print("no_img: %d" % no_img)
print("image_height: %d" % height)
print("image_width: %d" % width)


np.random.seed(798)

# Extracting the data i.e. each pic to X
X = df.data
no_pixels = X.shape[1]
print("features: %d" % no_pixels)
print('X_dataset_shape: {}'.format(X.shape[0]))

y = df.target
y.shape

# the label to predict is the id of the person
y = df.target
target_names = df.target_names
n_classes = target_names.shape[0]

print("no_of_classes: %d" % n_classes)
print(target_names)
print(y)


# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

print('Number of observations in the total set: {}'.format(no_img))
print('Number of observations in the training set: {}'.format(X_train.shape[0]))
print('Number of observations in the test set: {}'.format(X_test.shape[0]))

# Randomly selecting 200 PC's to explain the variance.
n_components = 200
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# Cumulative Variance Explained
var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)

# Finding the ideal number of components that explains atleast 95% of the data

arr = (pca.explained_variance_ratio_.cumsum())
ideal = np.abs(arr - 0.95).argmin()

# Substituting these values and transforning our data
n_components = ideal

pca = PCA(n_components=ideal, whiten=True).fit(X_train)

pcaface = pca.components_.reshape((n_components, height, width))

# After fitting, transform the X_train and X_test data.

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Training a SVM classification model
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

mod = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
mod = mod.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(mod.best_estimator_)

y_pred = mod.predict(X_test_pca)

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# plotting the result


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

n_row = 2
n_col = 6

pl.figure(figsize=(2 * n_col, 1.9 * n_row))
pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.29)
for i in range(n_row * n_col):
    pl.subplot(n_row, n_col, i + 1)
    pl.imshow(X_test[i].reshape((height, width)), cmap=pl.cm.gray)
    pl.title(prediction_titles[i], size=12)
    pl.xticks(())
    pl.yticks(())
pl.show()
