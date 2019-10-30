import numpy as np
import operator
from collections import Counter


class KNearestNeighbours:
    def __init__(self, k, weights):
        self.k = k
        self.weights = weights
        self.result = []
        self.result_w = []

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        print("Training done!")

    def predict(self, x_test):
        for j in x_test:
            distance = {}
            distance_w = {}
            counter = 0

            for i in self.x_train:
                sum = 0
                for k in range(len(j)):
                    sum += (j[k] - i[k]) ** 2

                if self.weights == "uniform":
                    distance[counter] = sum ** 1 / 2
                elif self.weights == "distance":
                    distance_w[counter] = 1 / (sum ** 1 / 2)
                counter += 1

            if self.weights == "uniform":
                distance = sorted(distance.items(), key=operator.itemgetter(1))
                self.result.append(self.classify(distance[:self.k]))
                del distance

            elif self.weights == "distance":
                distance_w = sorted(distance_w.items(), key=operator.itemgetter(1), reverse=True)
                self.result_w.append(self.classify2(distance_w[:self.k]))
                del distance_w

        if self.weights == "uniform":
            return self.result
        elif self.weights == "distance":
            return self.result_w

    def classify(self, distance):
        lable = []

        for i in distance:
            lable.append(self.y_train[i[0]])

        return Counter(lable).most_common()[0][0]

    def classify2(self, distance):
        y = np.unique(self.y_train)
        y_sum = [None]*len(y)

        for j in range(len(y)):
            sum = 0
            for i in distance:
                if y[j] == self.y_train[i[0]]:
                    sum += i[1]
            y_sum[j] = sum

        return y[y_sum.index(max(y_sum))]
