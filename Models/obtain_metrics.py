# +
'''
Defines the metrics used to evaluate each models' performance on the test set.
'''

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# -

def get_metrics(y_test, predictions, onehot_encoded=True):
    '''
    Calculates the precision, recall, and average f1 score on the test set.
    
    Params:
        - y_test: the labels from the testing set
        - predictions: the probabilities (or labels) predicted on the test set
        - onhot_encoded: true if labels are onehot encoded, false if integer encoded
    
    Returns:
        - Array of precision values for each class and recall values for each class and the average
        f1 score across all the classes.
    '''
    if onehot_encoded:
        y_test = y_test.argmax(axis=1)
        predictions = predictions.argmax(axis=1)
    
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)
    avg_f1 = f1_score(y_test, predictions, average='macro')
    
    return precision, recall, avg_f1

def plot_confusion_matrix(y_test, predictions, labels, title, onehot_encoded=True):
    '''
    Plots the confusion matrix.
    
    Params:
        - y_test: the labels from the testing set
        - predictions: the probabilities (or labels) predicted on the test set
        - labels: array of labels (strings) to use in the confusion matrix
        - title: title (string) of the confusion matrix
        - onhot_encoded: true if labels are onehot encoded, false if integer encoded
    '''
    if onehot_encoded:
        y_test = y_test.argmax(axis=1)
        predictions = predictions.argmax(axis=1)
    
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
    disp.ax_.set_title(title)
    disp.figure_.savefig('../Figures/' + title + '.png', bbox_inches='tight')
