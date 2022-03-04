import Vectors.vectors as vector

class Perceptron:
    # initlizing weigths
    def __init__(self, weigths: list, learning_rate):
        self.w = vector.Vector(weigths)
        self.learning_rate = learning_rate
        self.correct = 0
    # relu activtion function
    def relu(self, x):
        if x > 0:
            return x
        else:
            return -1
    # sigmoid activation function
    def sigmoid(self, x):
        return (3.7 ** x)/((3.7 ** x) + 1)
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
        if relu:
            return self.relu(answer)
        if sigmoid:
            return self.sigmoid(answer)
        return self.deflaut_activation(answer)
    # gradient descent
    def gradient_descent(self, correct, weigth, guess, input):
        error = correct - guess
        if error == 0:
            self.correct = self.correct + 1
            return weigth
        new_weigth = weigth + error * input * self.learning_rate
        return new_weigth
    # train the perceptron
    def train(self, inputs, answers, epochs, relu=False, sigmoid=False):
        for epoch in range(epochs):
            self.correct = 0
            i = 0
            for xs in inputs:
                guess = self.predict(inputs[i], relu=relu, sigmoid=sigmoid)
                for index in range(len(xs)):
                    self.w.inputs[index] = self.gradient_descent(answers[i], self.w.inputs[index], guess, xs[index])
                i = i + 1
            print(f"Epochs:{epoch} Percentage: {(self.correct/len(answers)) * 10}, Correct:{self.correct}")
    # test the perceptron
    def test(self, inputs, answers, relu=False, sigmoid=False):
        self.correct = 0
        i = 0
        for _ in inputs:
            guess = self.predict(inputs[i], relu=relu, sigmoid=sigmoid)
            if guess == answers[i]:
                self.correct = self.correct + 1
            i = i + 1
        print(f"Correct: {self.correct}")