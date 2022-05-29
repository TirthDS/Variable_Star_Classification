from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder, StandardScaler
from tensorflow.keras import models, layers, optimizers, regularizers
from sklearn.model_selection import train_test_split
from obtain_metrics import get_precision_recall, plot_confusion_matrix

def split_single_input(x_data, y_data, split=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x_data, test_size = split, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = (split / (1 - split)), random_state=0)
    
    return (x_train, y_train), (x_val, y_val), (x_train, y_val)

def split_triple_input(time_sorted_mags, phase_diffs, mag_diffs, y_data, split=0.2):
    x_seq_train, x_seq_test, y_train, y_test = train_test_split(time_sorted_mags, y_data, test_size = split, random_state=0)
    x_seq_train, x_seq_val, y_train, y_val = train_test_split(x_seq_train, y_train, test_size = (split / (1 - split)), random_state=0)

    x_phase_train, x_phase_test, _1, _2 = train_test_split(phase_diffs, y_data, test_size = split, random_state=0)
    x_phase_train, x_phase_val, _1, _3 = train_test_split(x_phase_train, _1, test_size = (split / (1 - split)), random_state=0)

    x_mag_train, x_mag_test, _4, _5 = train_test_split(mag_diffs, y_data, test_size = split, random_state=0)
    x_mag_train, x_mag_val, _4, _6 = train_test_split(x_mag_train, _4, test_size = (split / (1 - split)), random_state=0)
    
    return ([x_seq_train, x_phase_train, x_mag_train], y_train), ([x_seq_val, x_phase_val, x_mag_val], y_val), ([x_seq_test, x_phase_test, x_mag_test], y_test)

def triple_input_lstm_model(num_datapoints_1, num_datapoints_2, num_datapoints_3):
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
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    model.fit(train_data[0], train_data[1], 
              validation_data=val_data, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    loss, acc = model.evaluate(test_data[0], test_data[1])
    predictions = model.predict(test_data[0])
    
    return loss, acc, predictions

if __name__ == '__main__':
    # TODO
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    