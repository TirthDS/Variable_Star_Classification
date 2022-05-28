# PCA (DOES NOT WORK WELL, need > 50 Components to get good fit)

from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = train_test_split(x_data, onehot_encoded, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

scaler = StandardScaler()
normalized_train_X = scaler.fit_transform(X_train)
normalized_test_X = scaler.transform(X_test)
normalized_val_X = scaler.transform(X_val)

pca = PCA(n_components= 'mle', svd_solver = 'full')
pca.fit(normalized_train_X)
print(np.sum(pca.explained_variance_ratio_))

reduced_train_X = pca.transform(normalized_train_X)
reduced_test_X = pca.transform(normalized_test_X)
redcued_val_X = pca.transform(normalized_val_X)

clf = sklearn.linear_model.LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs', max_iter=100000)

clf.fit(reduced_train_X, y_train)

sm_score = clf.score(reduced_test_X, y_test)

sm_score

def base_model():
    X_input = layers.Input(445,)
    X = layers.Dense(128, activation='relu', kernel_regularizer = regularizers.l2(0.01))(X_input)
    X = layers.Dense(64, activation='relu')(X)
    X = layers.Dense(32, activation='relu')(X)
    X = layers.Dense(4, activation='softmax')(X)
    model = models.Model(inputs = X_input, outputs=X, name='something')
    
    return model

model = base_model()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(reduced_train_X, y_train, validation_data = (redcued_val_X, y_val), 
                    epochs = 20, batch_size=64, shuffle=True)

results = model.evaluate(reduced_test_X, y_test, batch_size=128)