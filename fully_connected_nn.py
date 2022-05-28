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