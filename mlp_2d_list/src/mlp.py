import random
from pandas_ml import ConfusionMatrix


class Mlp:
    def __init__(self, inputs, targets, nhidden):

        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0

        # Hidden layer weights with bias
        self.v = [[random.uniform(-1.0, 1.0)] * nhidden, [random.uniform(-1.0, 1.0)] * len(inputs[0] + 1)]
        self.v = np.random.uniform(low=-1.0, high=1.0, size=(nhidden, len(inputs[0]) + 1))
        # Delta update for adding momentum
        self.update_v = np.zeros(np.shape(self.v))
        # Output layer weights with bias
        self.w = np.random.uniform(low=-1.0, high=1.0, size=(8, nhidden + 1))
        # Delta update for adding momentum
        self.update_w = np.zeros(np.shape(self.w))
        # Hidden layer activation
        self.a = np.zeros(nhidden)
        # Output activation
        self.y = np.zeros(len(targets[0]))

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

        # Compute the error at the output
        delta_o = (target - self.y)
        # Compute the error in the hidden layer
        delta_h = self.a * (1 - self.a) * np.dot(self.w.transpose(), delta_o)[1:]

        # Update the output layer weights
        self.update_w = (np.outer(delta_o, np.insert(self.a, 0, -1.0))) + self.momentum * self.update_w
        # Update the hidden layer weights
        self.update_v = (np.outer(delta_h, np.insert(input, 0, -1.0))) + self.momentum * self.update_v

        self.w += self.update_w
        self.v += self.update_v

    def forward(self, inputs):
        """Computes the activation function from the hidden layer, to the output layer.

        :param {array-like}, shape = [n_features] inputs: Training vector with 'n_features' features/inputs.
        :return: None
        """
        # Compute the activation of each neuron j in the hidden layer
        for zeta in range(len(self.a)):
            # Compute sigma(weighted sum with bias)
            self.a[zeta] = self.sigmoid(np.dot(inputs, self.v[zeta][1:]) + self.v[zeta][0] * -1)
        # Compute the activation of each neuron j in the output layer
        for kappa in range(len(self.y)):
            # Compute sigma(weighted sum with bias)
            self.y[kappa] = self.sigmoid(np.dot(self.a.transpose(), self.w[kappa][1:]) + self.w[kappa][0] * -1)

    def confusion(self, inputs, targets):
        """Prints the confusion matrix
               :param inputs:
               :param targets:
               :return:
               """
        predicted = []
        expected = []

        # Produce confusion matrix arrays
        for i, t in zip(inputs, targets):
            self.forward(i)
            predicted.append(np.argmax(self.y))
            expected.append(np.argmax(t))

        confusion_m = ConfusionMatrix(expected, predicted)
        confusion_m.print_stats()

    def sigmoid(self, h):
        """Returns result from Sigmoid activation function g(h) = 1 / (1 + exp(-beta*h)

        :param float h: Weighted sum input
        :return : g(h)
        """
        return 1.0 / (1.0 + np.exp(-self.beta * h))

    def error(self, targets):
        """Computes the squared error sum

        :param {array-like}, shape = [n_outputs] targets: Target vector with 'n_outputs' targets
        :return: Squared error
        """
        err_sum = 0
        for t, y in zip(targets, self.y):
            err_sum += (y - t)

        return (err_sum ** 2) / 2
