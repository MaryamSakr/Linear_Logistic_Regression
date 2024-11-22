from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class LogisticRegressionModel:

    def __init__(self, x_train, y_train, x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def calc_accuracy(self, y_pred):
        accuracy = accuracy_score(y_pred, self.y_test)
        return accuracy

    def run_log(self):
        lg = SGDClassifier(max_iter=500, loss='log_loss')
        lg.fit(self.x_train,self.y_train)
        y_pred = lg.predict(self.x_test)
        return y_pred









