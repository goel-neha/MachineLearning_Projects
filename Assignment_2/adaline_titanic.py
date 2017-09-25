class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,
            # in the case of logistic regression, we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
fields = ['Survived','Pclass', 'Sex', 'Age']
df_train = pd.read_csv("train_preprocess.csv", usecols=fields)
#df_test = pd.read_csv("train_preprocess.csv", usecols=fields)
y = df_train.iloc[:, 0].values
X = df_train.iloc[:, [1, 2, 3]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print (y_train.shape)
print (X_train.shape)
print (y_test.shape)
print (X_test.shape)
#print (df_test.values)

ada1 = AdalineGD(n_iter=10, eta=0.0001)
ada1_fit = ada1.fit(X_train, y_train)
ax[0].plot(range(1, len(ada1_fit.cost_) + 1), np.log10(ada1_fit.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.0001')
# plt.show()
# ada2 = AdalineGD(n_iter=10, eta=0.0001)
# ada2_fit = ada2.fit(X_train, y_train)
# ax[1].plot(range(1, len(ada2_fit.cost_) + 1), ada2_fit.cost_, marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Sum-squared-error')
# ax[1].set_title('Adaline - Learning rate 0.0001')

plt.tight_layout()
# plt.savefig('./adaline_1.png', dpi=300)

y_pred = ada1.predict(X_test)
# print (y_pred)
no_of_correct = 0
temp_list=[]
for element in y_test:
    if element == 0:
        temp_list.append(-1)
    else:
        temp_list.append(1)
y_test = temp_list
for index, items in enumerate(y_pred):
    if items == y_test[index]:
        no_of_correct = no_of_correct + 1
accuracy = no_of_correct/len(y_pred) * 100.00
print (accuracy)

plt.show()
