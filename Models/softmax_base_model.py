#This code block is to import the data necessary (Lines 2-39)
import sklearn
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix
import numpy as np
import pickle
import pandas as pd

# import data
x_data = pd.read_pickle('extracted_features_200') #from tsfresh - up to 200 features
    
with open('data_labels_new', 'rb') as f:
    y_data = pickle.load(f)

x_data = x_data.to_numpy()

# integer encoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_data)

list(label_encoder.inverse_transform([0, 1, 2, 3]))

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# split data into train-valid-test
X_train, X_test, y_train, y_test = train_test_split(x_data, onehot_encoded, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state=1)

#normalized_train_X = normalize(X_train, axis=0)
scaler = StandardScaler()
normalized_train_X = scaler.fit_transform(X_train)
normalized_test_X = scaler.transform(X_test)
normalized_val_X = scaler.transform(X_val)


#Baseline Softmax Regression Model
import matplotlib.pyplot as plt
import sklearn

# Initialize Multinomial Regression
clf = sklearn.linear_model.LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs', max_iter=10000)

# fit model
clf.fit(normalized_train_X, np.ravel(y_train, order = 'C'))

# calculate model accuracy
sm_score = clf.score(normalized_test_X, y_test)
predictions = clf.predict(normalized_test_X)
precision_score(y_test, predictions, average=None)
recall_score(y_test, predictions, average=None)

# plot model and confusion matrix
fig = plt.figure()
cm = plot_confusion_matrix(clf, normalized_test_X, y_test)
plt.matshow(cm)
fig.savefig('confusion_softmax.png')