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


# Fully Connected Neural Network Model
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, metrics

X_train, X_test, y_train, y_test = train_test_split(x_data, onehot_encoded, test_size=0.2, random_state=1)
scaler = StandardScaler()
normalized_train_X = scaler.fit_transform(X_train)
normalized_test_X = scaler.transform(X_test)

def base_model():
    X_input = layers.Input(452,)
    X = layers.Dense(128, activation='relu', kernel_regularizer = regularizers.l2(0.01))(X_input)
    X = layers.Dense(64, activation='relu')(X)
    X = layers.Dense(32, activation='relu')(X)
    X = layers.Dense(4, activation='softmax')(X)
    model = models.Model(inputs = X_input, outputs=X, name='baseline')
    
    return model

model = base_model()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(normalized_train_X, y_train, validation_data = (normalized_val_X, y_val), epochs = 15, batch_size=64, shuffle=True)

results = model.evaluate(normalized_test_X, y_test)