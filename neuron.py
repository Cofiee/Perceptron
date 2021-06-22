import numpy as np


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y): #for each wektora "xi" oraz etykiety "target"  z odpowiednio zbioru wektorow cech "X" i wektora etykiet "x"
                update = self.eta * (target - self.predict(xi)) #delta w obliczane z n(yi - yxi) n - wspolczynnik uczenia # y - rzeczywista etykieta, yxi przewidziana etykita
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def eksport(self, filepath):
        f = open(filepath, "w")
        f.write(str(self.w_))
        f.close()
        return self

    def importWeights(self, filepath):
        f = open(filepath, "r")
        self.w_ = [float(x) for x in f.readline().split()]   #[-0.4, -0.68, 1.82]
        f.close()
        return self