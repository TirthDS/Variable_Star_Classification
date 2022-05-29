from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

def get_precision_recall(y_test, predictions, onehot_encoded=True):
    if onehot_encoded:
        y_test = y_test.argmax(axis=1)
        predictions = predictions.argmax(axis=1)
    
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, acerage=None)
    
    return precision, recall

def plot_confusion_matrix(y_test, predictions, labels, title, onehot_encoded=True):
    if onehot_encoded:
        y_test = y_test.argmax(axis=1)
        predictions = predictions.argmax(axis=1)
    
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
    disp.ax_.set_title(title)
    disp.figure_.savefig('../Figures/' + title + '.png', bbox_inches='tight')