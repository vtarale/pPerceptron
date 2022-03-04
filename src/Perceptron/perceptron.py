from xml.etree.ElementInclude import default_loader
import Vectors.vectors as vector
import math

class Perceptron:
    # initlizing weigths
    def __init__(self, weigths: list, learning_rate):
        self.w = vector.Vector(weigths)
        self.learning_rate = learning_rate
    # relu activtion function
    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0
    # sigmoid activation function
    def sigmoid(self, x):
        return (math.e ** x)/((math.e ** x) + 1)
    # the deflaut activation function
    def deflaut_activation(self, x):
        if x <= 0:
            return 0
        else:
            return 1
    # make a prediction
    def predict(self, x: list, relu=False, sigmoid=False):
        inputs = vector.Vector(x)
        answer = self.w.dot(inputs)
        if relu:
            return self.relu(answer)
        if sigmoid:
            return self.sigmoid(answer)
        return self.deflaut_activation(answer)
    # gradient descent
    def gradient_descent(self, correct, weigth, guess, input):
        error = correct - guess
        new_weigth = weigth + error * input
        return new_weigth
