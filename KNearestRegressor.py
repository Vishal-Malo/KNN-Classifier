import numpy as np
import operator


class KNearestRegressors:
    def __init__(self, k):
        self.k = k
        self.result = []
        self.result_w = []

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        print("Training done!")

    def predict(self, x_test):
        for j in x_test:
            distance = {}
            counter = 0

            for i in self.x_train:
                sum = 0
                for k in range(len(j)):
                    sum += (j[k] - i[k]) ** 2

                distance[counter] = sum ** 1 / 2
                counter += 1

            distance = sorted(distance.items(), key=operator.itemgetter(1))
            self.result.append(self.classify(distance[:self.k]))
            del distance

        return self.result

    def classify(self, distance):
        y = np.unique(self.y_train)
        y_sum = [None]*len(y)

        for j in range(len(y)):
            sum = 0
            for i in distance:
                if y[j] == self.y_train[i[0]]:
                    sum += i[1]
            y_sum[j] = sum/len(distance)

        return y[y_sum.index(max(y_sum))]
