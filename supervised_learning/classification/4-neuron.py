#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification."""

import numpy as np


class Neuron:
    """Neuron class that defines a single neuron for binary classification."""

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Parameters:
        nx (int): Number of input features.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """calculate forward propagation"""
        s = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.e ** (-s))
        return self.__A

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
root@b7769eee7ab34542a1d54c6ce6bd335c-2377118072:~/holbertonschool-machine_learning/supervised_learning/classification# ls
0-neuron.py  1-neuron.py  2-neuron.py  README.md
root@b7769eee7ab34542a1d54c6ce6bd335c-2377118072:~/holbertonschool-machine_learning/supervised_learning/classification# vi 3-neuron.py
root@b7769eee7ab34542a1d54c6ce6bd335c-2377118072:~/holbertonschool-machine_learning/supervised_learning/classification# git
[main 426600b] vf
 1 file changed, 50 insertions(+)
 create mode 100644 supervised_learning/classification/3-neuron.py
Enumerating objects: 8, done.
Counting objects: 100% (8/8), done.
Delta compression using up to 2 threads
Compressing objects: 100% (5/5), done.
Writing objects: 100% (5/5), 957 bytes | 957.00 KiB/s, done.
Total 5 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
To https://github.com/zeynabmirzazada/holbertonschool-machine_learning.git
   4e154bb..426600b  main -> main
root@b7769eee7ab34542a1d54c6ce6bd335c-2377118072:~/holbertonschool-machine_learning/supervised_learning/classification# cat 3-neuron.py
#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification."""

import numpy as np


class Neuron:
    """Neuron class that defines a single neuron for binary classification."""

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Parameters:
        nx (int): Number of input features.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """calculate forward propagation"""
        s = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.e ** (-s))
        return self.__A

    def cost(self, Y, A):
        '''logistic cost function'''
        return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        '''evaluation metrcs'''
        pred = self.forward_prop(X)
        pred = np.where(pred > 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return pred, cost

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
