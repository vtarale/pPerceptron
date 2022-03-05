import Vectors.vectors as vector
import math

class Perceptron:
    # initlizing weigths
    def __init__(self, weigths: list, learning_rate, bias):
        self.w = vector.Vector(weigths)
        self.learning_rate = learning_rate
        self.correct = 0
        self.bias = bias
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
                error = answers[i] - guess
                if error == 0:
                    self.correct = self.correct + 1
                else:
                    for index in range(len(xs)):
                        self.w.inputs[index] = self.w.inputs[index] + self.gradient_descent(error, xs[index])
                i = i + 1
            print(f"Epochs:{epoch}, Correct:{self.correct}")
    # test the perceptron
    def test(self, inputs, answers, relu=False, sigmoid=False):
        self.correct = 0
        for index in range(len(inputs)):
            guess = self.predict(inputs[index], relu=relu, sigmoid=sigmoid)
            if guess == answers[index]:
                self.correct = self.correct + 1
        print(f"Correct: {self.correct}")
