# +
'''
This file contains the methods to run the CNN-LSTM models using both
zero-padded/truncated data, and the phase folded data.
'''

from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder, StandardScaler
from tensorflow.keras import models, layers, optimizers, regularizers
from sklearn.model_selection import train_test_split
from obtain_metrics import get_metrics, plot_confusion_matrix


# -

def split_single_input(x_data, y_data, split=0.2):
    '''
    Split the input into train/dev/test sets for the single input CNN-LSTM 
    (only on the sequence data).
    
    Params:
        - x_data: zero-padded time sorted magnitude data (sequence data)
        - y_data: onehot encoded labels
    
    Returns:
        - Tuples of the train, dev, and test sets.
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_data, test_size = split, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = (split / (1 - split)), random_state=0)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def split_triple_input(time_sorted_mags, phase_diffs, mag_diffs, y_data, split=0.2):
    '''
    Split the three inputs into train/dev/test sets for the triple inpu CNN-LSTM
    (sequence data, phase differences, and phase-sorted magnitude differences).
    
    Params:
        - time_sorted_mags: time sorted magnitudes (zero-padded/truncated)
        - phase_diffs: phases with consecutive differences taken
        - mag_diffs: phase sorted magnitudes with consecutive differences taken
        
    Returns:
        - Tuples of the train, dev, and test sets.
    '''
    x_seq_train, x_seq_test, y_train, y_test = train_test_split(time_sorted_mags, y_data, test_size = split, random_state=0)
    x_seq_train, x_seq_val, y_train, y_val = train_test_split(x_seq_train, y_train, test_size = (split / (1 - split)), random_state=0)

    x_phase_train, x_phase_test, _1, _2 = train_test_split(phase_diffs, y_data, test_size = split, random_state=0)
    x_phase_train, x_phase_val, _1, _3 = train_test_split(x_phase_train, _1, test_size = (split / (1 - split)), random_state=0)

    x_mag_train, x_mag_test, _4, _5 = train_test_split(mag_diffs, y_data, test_size = split, random_state=0)
    x_mag_train, x_mag_val, _4, _6 = train_test_split(x_mag_train, _4, test_size = (split / (1 - split)), random_state=0)
    
    return ([x_seq_train, x_phase_train, x_mag_train], y_train), ([x_seq_val, x_phase_val, x_mag_val], y_val), ([x_seq_test, x_phase_test, x_mag_test], y_test)

def triple_input_lstm_model(num_datapoints_1, num_datapoints_2, num_datapoints_3):
    '''
    Defines the CNN-LSTM model for the triple input variation.
    
    Params:
        - num_datapoints_1: the number of datapoints per training example in the time-sorted magnitude data
        - num_datapoints_2: the number of datapoints per training example in the phase difference data
        - num_datapoints_3: the number of datapoints per training example in the phase-sorted magnitude difference
        data
        
    Returns:
        - Keras model object.
    '''
    input_1 = layers.Input((num_datapoints_1, 1))
    X1 = layers.Conv1D(64, 3, activation = 'relu')(input_1)
    X1 = layers.Conv1D(64, 3, activation = 'relu')(X1)
    X1 = layers.MaxPooling1D()(X1)
    X1 = layers.Conv1D(128, 3, activation = 'relu')(X1)
    X1 = layers.Conv1D(128, 3, activation = 'relu')(X1)
    X1 = layers.MaxPooling1D()(X1)
    X1 = layers.LSTM(128, return_sequences='True')(X1)
    
    input_2 = layers.Input((num_datapoints_2, 1))
    X2 = layers.Conv1D(64, 3, activation = 'relu')(input_2)
    X2 = layers.Conv1D(64, 3, activation = 'relu')(X2)
    X2 = layers.MaxPooling1D()(X2)
    X2 = layers.Conv1D(128, 3, activation = 'relu')(X2)
    X2 = layers.Conv1D(128, 3, activation = 'relu')(X2)
    X2 = layers.MaxPooling1D()(X2)
    
    input_3 = layers.Input((num_datapoints_3, 1))
    X3 = layers.Conv1D(64, 3, activation = 'relu')(input_3)
    X3 = layers.Conv1D(64, 3, activation = 'relu')(X3)
    X3 = layers.MaxPooling1D()(X3)
    X3 = layers.Conv1D(128, 3, activation = 'relu')(X3)
    X3 = layers.Conv1D(128, 3, activation = 'relu')(X3)
    X3 = layers.MaxPooling1D()(X3)
    X3 = layers.LSTM(128, return_sequences='True')(X3)
    
    
    merged = layers.Concatenate(axis=1)([X1, X2, X3])
    X = layers.Flatten()(merged)
    X = layers.Dense(256, activation = 'relu', kernel_regularizer = regularizers.L2(0.001))(X)
    X = layers.Dense(4, activation = 'softmax')(X)
    
    model = models.Model(inputs = [input_1, input_2, input_3], outputs = X, name='lstm')
    return model

def single_input_lstm_model(num_datapoints):
    '''
    Defines the baseline CNN-LSTM model for the single input variation.
    
    Params:
        - num_datapoints: the number of datapoints in each training example for the time-sorted
        magnitude sequential data.
    
    Returns:
        - Keras model object.
    '''
    
    X_input = layers.Input((num_datapoints, 1))
    X = layers.Conv1D(64, 3, activation = 'relu')(X_input)
    X = layers.Conv1D(64, 3, activation = 'relu')(X)
    X = layers.MaxPooling1D()(X)
    X = layers.Conv1D(128, 3, activation = 'relu')(X)
    X = layers.Conv1D(128, 3, activation = 'relu')(X)
    X = layers.MaxPooling1D()(X)
    X = layers.Bidirectional(layers.LSTM(64))(X)
    X = layers.Dense(256, activation = 'relu')(X)
    X = layers.Dense(4, activation = 'softmax')(X)
    
    model = models.Model(inputs = X_input, outputs = X, name='base_lstm')
    return model

def fit_model(model, train_data, val_data, test_data, epochs=30, batch_size=64):
    '''
    Runs batch gradient descent to fit the model to the training data and generates predictions
    on the test dataset.
    
    Params:
        - model: the Keras model to train on
        - train_data: tuple containing the x training set and y training labels
        - val_data: tuple containing the x validation set and y validation labels
        - test_data: tuple containing the x testing set and y testing set
        - epochs: the number of epochs to train over
        - batch_size: the size of batches to use during batch GD
    
    Returns:
        - loss and accuracy on the testing set, and predictions on the testing set
    '''
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    model.fit(train_data[0], train_data[1], 
              validation_data=val_data, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    loss, acc = model.evaluate(test_data[0], test_data[1])
    predictions = model.predict(test_data[0])
    
    return loss, acc, predictions

if __name__ == '__main__':
    with open('../Data/cnn_lstm_preprocessing/padded_sequence_data', 'rb') as f:
        padded_sequence_data = np.array(pickle.load(f))
        
    with open('../Data/cnn_lstm_preprocessing/diff_phase_folded_data', 'rb') as f:
        diff_phase_folded_data = np.array(pickle.load(f))

    with open('../Data/data_labels', 'rb') as f:
        y_data = np.array(pickle.load(f))
        
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    # Run Base CNN-LSTM
    train1, val1, test1 = split_single_input(padded_sequence_data, onehot_encoded)
    base_model = single_input_lstm_model(train1[0].shape[1])
    loss_base, acc_base, predictions_base = fit_model(base_model, train1, val1, test1)
    print('Base CNN-LSTM:')
    precision_base, recall_base, f1_base = get_metrics(test1[1], predictions_based)
    print('Precision: ' + str(precision_base))
    print('Recall: ' + str(recall_base))
    print('Average F1 Score: ' + str(f1_base)) 
    plot_confusion_matrix(test1[1], predictions_based, 'Base CNN LSTM CM')
    
    
    # Run triple input CNN-LSTM
    phase_differences = padded_sequence_data[:, 0]
    mag_differences = padded_sequence_data[:, 1]
    
    train2, val2, test2 = split_triple_input(padded_sequence_data, phase_differenes, mag_differences, onehot_encoded)
    model = triple_input_lstm_model(train2[0][0].shape[1], train2[0][1].shape[1], train_2[0][2].shape[1])
    loss_triple, acc_triple, predictions_triple = fit_model(model, train2, val2, test2, epochs=10)
    print('Triple Input CNN-LSTM:')
    precision, recall, f1 = get_metrics(test2[1], predictions_triple)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('Average F1 Score: ' + str(f1)) 
    plot_confusion_matrix(test2[1], predictions_triple, 'Triple Input CNN LSTM CM')

    
