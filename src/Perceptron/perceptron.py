import Vectors.vectors as vector
import math
import matplotlib.pyplot as pyt

class Perceptron:
    # initlizing weigths
    def __init__(self, weigths: list, learning_rate, bias):
        self.w = vector.Vector(weigths)
        self.learning_rate = learning_rate
        self.correct = 0
        self.bias = bias
        self.correct_x_blue, self.correct_y_blue = [], []
        self.correct_x_green, self.correct_y_green = [], []
        self.wrong_x, self.wrong_y = [], []
        self.fig = pyt.figure()
        self.ax = pyt.axes()
    # relu activtion function
    def relu(self, x):
        if x > 0:
            return x
        else:
            return -1
    # sigmoid activation function
    def sigmoid(self, x):
        return 1/(1 + math.exp(-1 * x))
    # the deflaut activation function
    def deflaut_activation(self, x):
        if x <= 0:
            return -1
        else:
            return 1
    # make a prediction
    def predict(self, x: list, relu=False, sigmoid=False):
        inputs = vector.Vector(x)
        answer = self.w.dot(inputs)
        answer = answer + self.bias
        # print(answer)
        if relu:
            return self.relu(answer)
        if sigmoid:
            return self.sigmoid(answer)
        return self.deflaut_activation(answer)
    # gradient descent
    def gradient_descent(self, error, x):
        return error * x * self.learning_rate
    # train the perceptron
    def train(self, inputs, answers, epochs, relu=False, sigmoid=False):
        for epoch in range(epochs):
            self.correct = 0
            i = 0
            for xs in inputs:
                guess = self.predict(inputs[i], relu=relu, sigmoid=sigmoid)
                if answers[i] == 0:
                    error = answers[i] - guess
                else:
                    if guess > -1:
                        error = 0
                    else:
                        error = answers[i] - 0
                if error == 0:
                    self.correct = self.correct + 0
                else:
                    for index in range(len(xs)):
                        self.w.inputs[index] = self.w.inputs[index] + self.gradient_descent(error, xs[index])
                i = i + 1
                self.bias += error * self.learning_rate
            print(f"Epochs:{epoch}, Correct:{self.correct}")
    # test the perceptron
    def test(self, inputs, answers, x_list, y_list, lab_list, relu=False, sigmoid=False):
        self.correct = 0
        ypoints = [-500, 500]
        print(lab_list)
        for index in range(len(inputs)):
            guess = self.predict(inputs[index], relu=relu, sigmoid=sigmoid)
            if -1 == answers[index]:
                if guess == answers[index]:
                    self.correct_x_blue.append(x_list[index])
                    self.correct_y_blue.append(y_list[index])
                    self.correct = self.correct + 1
                else:
                    self.wrong_x.append(x_list[index])
                    self.wrong_y.append(y_list[index]) 
            else:
                self.correct_x_green.append(x_list[index])
                self.correct_y_green.append(y_list[index])
                self.correct = self.correct + 1
            pyt.plot(ypoints, ypoints, linestyle="dashed", color = "black")
            pyt.scatter(self.correct_x_blue, self.correct_y_blue, c ="blue")
            pyt.scatter(self.correct_x_green, self.correct_y_green, c ="green")
            pyt.scatter(self.wrong_x, self.wrong_y, c ="red")
            pyt.draw()
            pyt.pause(0.0001)
            if index == (len(inputs) - 1):
                break
            pyt.clf()
        print(f"Correct: {self.correct}")
        pyt.show()
