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

fig = plt.figure()
cm = plot_confusion_matrix(clf, normalized_test_X, y_test)
plt.matshow(cm)
fig.savefig('confusion_softmax.png')