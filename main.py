import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
#from neuron import Perceptron
#from neuronADALINE import Adaline
from AdalineSGD import  Adaline

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers=('s', 'x', 'o', '^', 'v')
    colors=('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)

df.tail()
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

#plt.plot(x, 0.22 + 0.37 * x)
#plt.show()
"""
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,
         marker='o')
plt.xlabel('Epoki')
plt.ylabel('Liczba aktualizacji')
plt.show()

X = [1.0, 0.5]
print(ppn.predict(X))
"""



ada1 = Adaline(n_iter=15, eta=0.01).fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada1)
plt.title('Adaline - gradient prosty')
plt.xlabel('Długość działki [standaryzowana]')
plt.ylabel('Długość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
ax[0].set_xlabel('Długość działki [standaryzowana]')
ax[0].set_ylabel('Długość płatka [standaryzowana]')
ax[0].set_title('Adaline - Współczynnik uczenia 0,01')

ax[1].plot(range(1, len(ada1.cost_) + 1),
           ada1.cost_, marker='o')
ax[1].set_xlabel('Epoki')
ax[1].set_ylabel('Suma kwadratow bledow')
ax[1].set_title('Adaline - Współczynnik uczenia 0,01')
plt.show()
