import  operator
from collections import Counter

class KNearestNeighbours:
    def __init__(self,k):
        self.k = k
        self.result = []

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        print("Tranning done!")

    def predict(self, x_test):

        for j in x_test:
            distance = {}
            counter = 0

            for i in self.x_train:
                sum = 0
                for k in range(len(j)):
                    sum += (j[k] - i[k])**2

                distance[counter] = (sum)**1/2
                counter += 1

            distance = sorted(distance.items(), key=operator.itemgetter(1))
            self.result.append(self.classify(distance[:self.k]))
            del(distance)

        return  self.result


    def classify(self,distance):
        label = []

        for i in distance:
            label.append(self.y_train[i[0]])


        return Counter(label).most_common()[0][0]


