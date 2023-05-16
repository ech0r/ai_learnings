#!/usr/bin/python3

import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def calculate_loss(self):
        return np.mean(np.square(self.y - self.output)) # MSE

# The XOR function
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(inputs, outputs)

for i in range(1500):
    nn.feedforward()
    nn.backprop()
    if i % 100 == 0:
        print(f"for iteration {i} the loss is {nn.calculate_loss()}")

print(nn.output)
