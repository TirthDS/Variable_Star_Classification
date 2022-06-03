# +
'''
Defines the methods used for classification using the tsfresh
extracted features as input (softmax and base neural network).
'''

import sklearn
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd
# -

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, metrics, optimizers
from obtain_metrics import get_metrics, plot_confusion_matrix

def load_raw_data(x_data_file, y_data_file):
    '''
    Retrieves the data stored as pickle files.
    
    Params:
        - x_data_file: the pickle file containing the extracted features
        - y_data_file: the pickle file containing the labels
    '''
    x_data_raw = np.array(pd.read_pickle(x_data_file))
    with open('y_data_file', 'rb') as f:
        y_data_raw = np.array(pickle.load(f))
        
    return x_data_raw, y_data_raw

def preprocess(x_data, y_data, split=0.2, one_hot=True):
    '''
    Preprocesses the data by generating onehot encodings for the labels,
    splitting them into train/dev/test sets, and normalizing feature-wise.
    
    Params:
        - x_data: the feature-extracted data
        - y_data: the onehot-encoded labels
        - split: the fraction to split the data into (defaults to 60/20/20)
        
    Returns:
        - Tuples for the train/validation/test set.
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
    Performs softmax classification on the training data and generates
    predictions on the testing data.
    
    Params:
        - train_data: tuple of the normalized feature-extracted data and onehot encoded labels (for training)
        - test_data: tuple of the normalized feature-extrated data and onhot encoded labels (for testing)
        
    Returns:
        - testing accuracy and predictions on the test set
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
    Defines a 4 layer fully connected neural network.
    
    Params:
        - num_features: the number of features extracted from tsfresh in the data
    
    Returns:
        - Keras model object.
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
    Compiles the model defined above and traings the model on the training data.
    Produces predictions on the testing data.
    
    Params:
        - train_data: tuple of the x and y data for the training set
        - val_data: tuple of the x and y data for the dev set
        - test_data: tuple of the x and y data for the test set
    
    Returns:
        - loss, accuracy, and predictions on the testing set
    '''
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data
    
    model = baseline_neural_network_model(x_train.shape[1])
    model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
                  loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit(x_train, y_train, validation_data = (x_val, y_val),
             epochs = epochs, batch_size=batch_size, shuffle=True)
    
    loss, acc = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)
    return loss, acc, predictions

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
    labels = ['eb', 'irreg', 'mira', 'rr']
    print('Softmax Classifier:')
    precision_sm, recall_sm, f1_sm = get_metrics(test_data_sm[1], sm_predictions, onehot_encoded=False)
    print('Precisions: ' + str(precision))
    print('Recall: ' + str(recall))
    print('Average F1 Score: ' + str(f1_sm))
    plot_confusion_matrix(test_data_sm[1], sm_predictions, labels, 'Softmax CM', onehot_encoded=False)
    
    # Preprocess to normalize and split (nn)
    train_data_nn, val_data_nn, test_data_nn = preprocess(x_data_raw, y_data_raw)
    
    # Train and predict with nn
    nn_loss, nn_acc, nn_predictions = fit_nn(train_data_nn, val_data_nn, test_data_nn)
    
    # Compute precision/recall/confusion matrix
    print('Fully Connected Neural Network:')
    precision_nn, recall_nn, f1_nn = get_metrics(test_data_nn[1], nn_predictions)
    print('Precisions: ' + str(precision))
    print('Recall: ' + str(recall))
    print('Average F1 Score: ' + str(f1_nn))
    plot_confusion_matrix(test_data_nn[1], nn_predictions, labels, 'FC Neural Network CM')
