import random
from math import exp
from pandas_ml import ConfusionMatrix


class Mlp:
    def __init__(self, inputs, targets, nhidden):

        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0

        # Hidden layer weights with bias
        self.v = [[random.uniform(-1.0, 1.0) for _ in range(len(inputs[0]) + 1)] for _ in range(nhidden)]
        # Delta update for adding momentum
        self.update_v = [[0] * len(self.v[0])] * nhidden
        # Output layer weights with bias
        self.w = [[random.uniform(-1.0, 1.0) for _ in range(nhidden + 1)] for _ in range(len(targets[0]))]
        # Delta update for adding momentum
        self.update_w = [[0] * nhidden] * len(targets[0])
        # Hidden layer activation
        self.a = [0] * nhidden
        # Output activation
        self.y = [0] * len(targets[0])

    def earlystopping(self, inputs, targets, valid, validtargets):
        """Runs the algorithm, for every training epoch: computes the total sum of squared error from the validation set
        and stops training if the error starts increasing.

        :param {array-like}, shape = [n_sequences][n_features] inputs: Training vectors with 'n_features' features/inputs.
        :param {array-like}, target = [n_sequences][n_targets] targets: Target vectors with 'n_targets' targets/outputs
        :param {array-like}, shape = [n_sequences][n_features] valid: Validation vectors with 'n_features' features/inputs.
        :param {array-like}, target = [n_sequences][n_targets] validtargets: Validation target vectors with 'n_features' features/inputs.
        :return:
        """

        fit = False
        error_sum = float("inf")
        print("Training network..")
        while not fit:
            errors = []
            self.train(inputs, targets)
            for j, t in zip(valid, validtargets):
                self.forward(j)
                errors.append(self.error(t))
            print("Error sum from validation set = ", sum(errors))
            if sum(errors) > error_sum:
                print("Error increasing, stopping training", sum(errors))
                fit = True
            error_sum = sum(errors)

    def train(self, inputs, targets, iterations=1):
        """ Runs forward and backwards phase of the backpropagation algorithm with a minibatch style training routine

        :param {array-like}, shape = [n_sequences][n_features] inputs: Training vectors with 'n_features' features/inputs.
        :param {array-like}, target = [n_sequences][n_targets] targets: Target vectors with 'n_targets' targets/outputs
        :param iterations: Number of iterations to run the minibatch routine.
        :return: None
        """

        # Repeat training iterations*"n" input vectors
        for _ in range(iterations * len(inputs)):
            rand_i = random.randint(0, len(inputs) - 1)
            # Run forwards phase
            self.forward(inputs[rand_i])
            # Run backwards phase
            self.backward(inputs[rand_i], targets[rand_i])

        return self

    def backward(self, input, target):
        """Computes the error at the output and the hidden layer, and updates the weights at output and hidden layer.

        :param {array-like}, shape = [n_features] input: Training vector with 'n_features' features/inputs.
        :param {array-like}, shape = [n_targets] target: Target vector with 'n_target' targets
        :return: None
        """

        # TODO: Control computation
        # Compute the error at the output
        delta_o = [0] * len(self.y)
        for i in range(len(delta_o)):
            delta_o[i] = (target[i] - self.y[i]) * self.y[i] * (1 - self.y[i])

        # TODO: Control computation
        delta_h = [0] * len(self.a)
        # Compute the weight * output error sum for the error in the hidden layer
        # Compute the error in the hidden layer
        for i in range(len(delta_h)):
            # Indexing weight +1 to avoid bias
            delta_h[i] = self.a[i] * (1 - self.a[i]) * sum(self.w[k][i + 1] * delta_o[k] for k in range(len(delta_o)))

        # TODO: Fix weight updates
        # Update the output layer weights
        for k in range(len(self.update_w)):
            for z in range(len(self.update_w[0])):
                self.update_w[k][z] = -self.eta * delta_o[k] * self.a[z] + self.momentum * self.update_w[k][z]
                self.w[k][z] += self.update_w[k][z]

        # Update the hidden layer weights
        bias_input = [-1.0] + input.tolist()
        for k in range(len(self.update_v)):
            for i in range(len(self.update_v[0])):
                self.update_v[k][i] = -self.eta * delta_h[k] * bias_input[i] + self.momentum * self.update_v[k][i]
                self.v[k][i] += self.update_v[k][i]

    def forward(self, inputs):
        """Computes the activation function from the hidden layer, to the output layer.

        :param {array-like}, shape = [n_features] inputs: Training vector with 'n_features' features/inputs.
        :return: None
        """
        # TODO: Control computation
        # Compute the activation of each neuron j in the hidden layer
        for z in range(len(self.a)):
            # Compute weighted sum with bias
            self.a[z] = self.sigmoid(sum(inputs[i] * self.v[z][i] for i in range(len(inputs))))
        # TODO: Control computation
        # Compute the activation of each neuron j in the output layer
        for k in range(len(self.y)):
            # Compute sigma(weighted sum with bias)
            self.y[k] = self.sigmoid(sum(self.a[j] * self.w[k][j] for j in range(len(self.a))))

    def confusion(self, inputs, targets):
        """Prints the confusion matrix
       :param inputs:
       :param targets:
       :return:
       """
        predicted = []
        expected = []

        # Produce confusion matrix arrays
        for i, t in zip(inputs, targets.tolist()):
            self.forward(i)
            predicted.append(self.y.index(max(self.y)))
            expected.append(t.index(max(t)))

        confusion_m = ConfusionMatrix(expected, predicted)
        confusion_m.print_stats()

    def sigmoid(self, h):
        """Returns result from Sigmoid activation function g(h) = 1 / (1 + exp(-beta*h)

        :param float h: Weighted sum input
        :return : g(h)
        """
        return 1.0 / (1.0 + exp(-self.beta * h))

    def error(self, targets):
        """Computes the squared error sum

        :param {array-like}, shape = [n_outputs] targets: Target vector with 'n_outputs' targets
        :return: Squared error
        """
        err_sum = 0
        for t, y in zip(targets, self.y):
            err_sum += (y - t)

        return (err_sum ** 2) / 2
