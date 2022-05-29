import sklearn
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import numpy as np
import pickle
import pandas as pd

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, metrics, optimizers

def load_raw_data(x_data_file, y_data_file):
    '''
    '''
    x_data_raw = np.array(pd.read_pickle(x_data_file))
    with open('y_data_file', 'rb') as f:
        y_data_raw = np.array(pickle.load(f))
        
    return x_data_raw, y_data_raw
    
def preprocess(x_data, y_data, split=0.2, one_hot=True):
    '''
    '''
    # Onehot encoding
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)
    
    if (one_hot):
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = y_data.reshape(len(integer_encoded), 1)
        y_data = onehot_encoder.fit_transform(integer_encoded)

    # Split into train/dev/test sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=split, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=(split / (1-split)), random_state=0)
    
    # Normalize feature-wise
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def softmax_classifier(train_data, test_data):
    '''
    '''
    clf = sklearn.linear_model.LogisticRegression(class_weight='balanced', multi_class='multinomial', 
                                                  solver='lbfgs', max_iter=10000)
    
    x_train, y_train = train_data
    x_test, y_test = test_data
    clf.fit(x_train, np.ravel(y_train, order='C'))
    
    sm_score = clf.score(x_test, y_test)
    predictions = clf.predict(x_test)
    return sm_score, predictions

def baseline_neural_network_model(num_features):
    '''
    '''
    X_input = layers.Input(num_features,)
    X = layers.Dense(128, activation='relu', kernel_regularizer = regularizers.l2(0.01))(X_input)
    X = layers.Dense(64, activation='relu')(X)
    X = layers.Dense(32, activation='relu')(X)
    X = layers.Dense(4, activation='softmax')(X)
    model = models.Model(inputs = X_input, outputs=X, name='baseline')
    return model

def fit_nn(train_data, val_data, test_data, epochs=15, batch_size=64):
    '''
    '''
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data
    
    model = baseline_model()
    model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
                  loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit(x_train, y_train, validation_data = (x_val, y_val),
             epochs = epochs, batch_size=batch_size, shuffle=True)
    
    loss, acc = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)
    return loss, acc, predictions

def decision_tree_classifier(train_data, test_data):
    '''
    TODO
    '''
    return

if __name__ == '__main__':
    # Load raw data (post feature extraction with tsfresh)
    extracted_features_file = '../data/feature_extraction/extracted_features_200'
    y_labels = '../data/data_labels'
    x_data_raw, y_data_raw = load_raw_data(extracted_features_file, y_labels)
    
    # Preprocess to normalize and split (softmax)
    train_data_sm, _, test_data_sm = preprocess(x_data_raw, y_data_raw, one_hot=False)
    
    # Train and predict with softmax classifier
    sm_acc, sm_predictions = softmax_classifier(train_data_sm, test_data_sm)
    
    # Compute precision/recall/confusion matrix
    print('Softmax Classifier:')
    # TODO
    
    # Preprocess to normalize and split (nn)
    train_data_nn, val_data_nn, test_data_nn = preprocess(x_data_raw, y_data_raw)
    
    # Train and predict with nn
    nn_loss, nn_acc, nn_predictions = fit_nn(train_data_nn, val_data_nn, test_data_nn)
    
    # Compute precision/recall/confusion matrix
    print('Fully Connected Neural Network:')
    # TODO