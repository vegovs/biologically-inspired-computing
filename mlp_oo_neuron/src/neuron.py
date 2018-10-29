import random
from math import exp


class Neuron(object):
    def __init__(self, index, eta=0.1, beta=1, n_inputs=10, hidden=False):
        """Initialization function for neuron class.
        :param float eta: Learning rate
        :param float beta: Activation function constant
        :param int n_inputs: Number of inputs/weights for the neuron.
        :param bool hidden: If in hidden layer or output layer.
        :param int index: Index in layer
        """
        self.eta = eta
        self.beta = beta
        self.n_inputs = n_inputs
        self.hidden = hidden
        self.index = index
        self.alpha = 0.1

        # Weights
        self.w = [random.uniform(-1.0, 1.0) for _ in range(n_inputs + 1)]
        # Weight updates used for momentum
        self.w_update = [0] * (n_inputs + 1)
        # Activation value
        self.y = 0
        # Error value
        self.d = 0

    def forward(self, x):
        """Computes the activation of the neuron.

        :param {array-like}, shape = [n_features] x: Training vector with 'n_features' features/inputs.
        :return: Self
        """
        self.y = self._g(self._h(x))
        return self

    def _h(self, x):
        """Net-input passed on to sigmoid function.

        :param {array-like}, shape = [n_features] x: Input vector with 'n_features' features/inputs.
        :return: Net-input
        """
        h = 0
        for xi, wi in zip(x, self.w[1:]):
            h += xi * wi
        # Add bias
        h += -1 * self.w[0]
        return h

    def _g(self, h):
        """Returns result from Sigmoid activation function g(h) = 1 / (1 + exp(-beta*h)

        :param float h: Weighted sum input
        :return : g(h)
        """
        return 1.0 / (1.0 + exp(-self.beta * h))

    def _d(self, K):
        """Computes the error delta(k) = (y-t)y(1-y) // delta(z) = a(1-a)sum(w_z*d(k))

        :param list K: Target if output neuron, or list of output neurons if hidden
        :return: Self
        """
        if self.hidden:
            # Sum of the output-delta*corresponding weight
            sum_wd = 0
            # For every output neuron k
            for k in range(len(K)):
                # Indexing weights +1 to avoid bias weight
                sum_wd += K[k].d*K[k].w[self.index + 1]
            self.d = self.y * (1 - self.y) * sum_wd
        else:
            self.d = (self.y - K) * self.y * (1 - self.y)
        return self

    def update_w(self, ax):
        """Update neuron weights with momentum using w <- w-eta*error*ax + alpha*w_update

        :param ax: Weight input(i.e for output layer: activation from previous hidden layer
        :return: Self
        """

        # Update bias weight
        self.w_update[0] = (- self.eta * self.d * self.w[0]) + (self.alpha * self.w_update[0])
        self.w[0] += self.w_update[0]
        # Update rest of weights
        for j in range(1, (len(self.w))):
            self.w_update[j] = (- self.eta * self.d * ax[j-1]) + (self.alpha * self.w_update[j])
            self.w[j] += self.w_update[j]

        return self
