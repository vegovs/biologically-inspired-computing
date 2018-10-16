import random
from pandas_ml import ConfusionMatrix

from src.neuron import Neuron


class Mlp:
    def __init__(self, inputs, n_hidden):
        """Initializes the Multilayer Neural Perceptron
        :param {array-like}, shape = [n_sequences][n_features] inputs: Training vectors with 'n_features' features/inputs.
        :param int n_hidden: Number of neurons in hidden layer.
        """
        self.beta = 1
        self.eta = 0.5

        # 8 classification values
        self.output = [Neuron(i, self.eta, self.beta, n_hidden, False) for i in range(8)]
        self.hidden = [Neuron(i, self.eta, self.beta, len(inputs[0]), True) for i in range(n_hidden)]

    def earlystopping(self, inputs, targets, valid, validtargets):
        """Runs the algorithm, for every training epoch: computes the total sum of squared error from the validation set
        and stops training if the error starts increasing.

        :param {array-like}, shape = [n_sequences][n_features] inputs: Training vectors with 'n_features' features/inputs.
        :param {array-like}, target = [n_sequences][n_targets] targets: Target vectors with 'n_targets' targets/outputs
        :param {array-like}, shape = [n_sequences][n_features] valid: Validation vectors with 'n_features' features/inputs.
        :param {array-like}, target = [n_sequences][n_targets] validtargets: Validation target vectors with 'n_features' features/inputs.
        :return:
        """
        i = 0
        fit = False
        # Array for storing total error sum every time validation set is run
        error_sum = [float("inf")]
        while not fit:
            # Array for storing errors from each validation vector
            errors = []
            # Train the mlp
            self._train(inputs, targets)
            # Compute error sum for validation set
            for j, t in zip(valid, validtargets):
                self._forward(j)
                errors.append(self._error(t))
            i += 1
            error_sum.append(sum(errors))
            print("Error sum from validation set: ", error_sum[i])
            if error_sum[i] > error_sum[i - 1]:
                print("Error increasing: ", error_sum[i], ">", error_sum[i], ", stopping training.")
                fit = True

    def _train(self, inputs, targets, iterations=2):
        """ Runs forward and backwards phase of the backpropagation algorithm

        :param {array-like}, shape = [n_sequences][n_features] inputs: Training vectors with 'n_features' features/inputs.
        :param {array-like}, target = [n_sequences][n_targets] targets: Target vectors with 'n_targets' targets/outputs
        :param iterations: Number of iterations to run the backpropagation algorithm.
        :return: None
        """

        # Repeat training iterations*"n" input vectors
        for _ in range(iterations*len(inputs)):
            rand_i = random.randint(0, len(inputs)-1)
            # Run forwards phase
            self._forward(inputs[rand_i])
            # Run backwards phase
            self._backward(targets[rand_i], inputs[rand_i])

        return self

    def _backward(self, target, inputs):
        """Computes the error at the output and the hidden layer, and updates the weights at output and hidden layer.

        :param {array-like}, shape = [n_targets] target: Target vector with 'n_target' targets
        :param {array-like}, shape = [n_features] inputs: Training vector with 'n_features' features/inputs.
        :return: None
        """

        # Computes the error at the output: delta(k) = (y-t)y(1-y)
        for k, tk in zip(self.output, target):
            k._d(tk)
        # Computes the error at the hidden layer: delta(z) = a(1-a)sum(w_z*d(k))
        for z in self.hidden:
            z._d(self.output)
        # Update output layer weights using w <- w-eta*error*weight_input + alpha*w
        for k in self.output:
            k.update_w([z.y for z in self.hidden])
        # Update hidden layer weights using w <- w-eta*error*weight_input + alpha*w
        for k in self.hidden:
            k.update_w(inputs)

    def _forward(self, inputs):
        """Computes the activation function from the hidden layer, to the output layer.

        :param {array-like}, shape = [n_features] inputs: Training vector with 'n_features' features/inputs.
        :return: None
        """
        # Compute the activation of each neuron j in the hidden layer
        for j in self.hidden:
            j.forward(inputs)
        # Compute the activation of each neuron k in the output layer
        for k in self.output:
            k.forward([y.y for y in self.hidden])

    def _error(self, targets):
        """Computes the squared error sum

        :param {array-like}, shape = [n_outputs] targets: Target vector with 'n_outputs' targets
        :return: Squared error
        """
        err_sum = 0
        for t, k in zip(targets, self.output):
            err_sum += (k.y - t)

        return (err_sum**2)/2

    def confusion(self, inputs, targets):
        """Prints the confusion matrix
        :param inputs:
        :param targets:
        :return:
        """
        predicted = []
        expected = []

        for i, t in zip(inputs, targets.tolist()):
            self._forward(i)
            out = self._transform_output()
            predicted.append(out.index(max(out)))
            expected.append(t.index(max(t)))

        confusion_m = ConfusionMatrix(expected, predicted)
        confusion_m.print_stats()

    def _transform_output(self):
        """Transforms the output to a argmax style list. Only largest number is 1, rest is 0

        :return:
        """
        output = [k.y for k in self.output]
        trans_out = [0] * len(self.output)
        trans_out[output.index(max(output))] = 1
        return trans_out
