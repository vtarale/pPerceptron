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
        self.brain = p.Perceptron([-1, 2], learning_rate, bias)

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
        self.x = []
        self.y = []
        self.label = []
        for index in range(r):
            self.train.append([self.points[index].x, self.points[index].y])
            self.answers_train.append(self.points[index].lab)
        for index in range(len(self.points) - r):
            self.x.append(self.points[index].x)
            self.y.append(self.points[index].y)
            self.label.append(self.points[index].lab)
            self.test.append([self.points[index].x, self.points[index].y])
            self.answers_test.append(self.points[index].lab)
    # train the perceptron
    def t(self, epochs, relu=False):
        self.brain.train(self.train, self.answers_train, epochs, relu=relu)
    # test the perceptron
    def te(self, relu=False):
        self.brain.test(self.test, self.answers_test, self.x, self.y, self.label, relu=relu)

if __name__ == "__main__":
    perceptron = Test(2.5, 2)
    perceptron.create_normal(1000, 500, 500)
    perceptron.split(750)
    perceptron.t(0, relu=True)
    perceptron.te(relu=True)   
