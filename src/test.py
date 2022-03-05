import random
import Perceptron.perceptron as p

class Point:
    def __init__(self, x, y, lab):
        self.x, self.y, self.lab = x, y, lab

class Test:
    def __init__(self, learning_rate, bias):
        self.points = []
        self.train = []
        self.test = []
        self.answers_train = []
        self.answers_test = []
        self.brain = p.Perceptron([5, 2], learning_rate, bias)

    def create_normal(self, points: int, rangex, rangey):
        for _ in range(points):
            x = random.randint(rangex * -1, rangex)
            y = random.randint(rangey * -1, rangey)
            label = 1
            if x < y:
                label = -1
            self.points.append(Point(x, y, label))
    # split training and testing data
    def split(self, r: int):
        for index in range(r):
            self.train.append([self.points[index].x, self.points[index].y])
            self.answers_train.append(self.points[index].lab)
        for index in range(len(self.points) - r):
            self.test.append([self.points[index].x, self.points[index].y])
            self.answers_test.append(self.points[index].lab)
    # train the perceptron
    def t(self, epochs, relu=False, sigmoid=False):
        self.brain.train(self.train, self.answers_train, epochs, relu=relu, sigmoid=sigmoid)
    # test the perceptron
    def te(self, relu=False, sigmoid=False):
        self.brain.test(self.test, self.answers_test, relu=relu, sigmoid=sigmoid)

if __name__ == "__main__":
    perceptron = Test(1, 1)
    perceptron.create_normal(30000, 500000, 5000000)
    perceptron.split(29900)
    perceptron.t(5)
    perceptron.te()