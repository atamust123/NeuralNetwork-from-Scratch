import numpy as np


class NeuralNetwork:
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivation(self, x):
        return x * (1 - x)

    def ReLu(self, x):
        x[x < 0] = 0
        return x

    def ReLu_derivation(self, x):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x

    def mse(self, error):
        return np.sum(np.square(error)) / len(error)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def neg_log_likelihood(self, output, y):  # cross entropy
        r = ((y * np.log10(output) + (1 - y) * np.log10(1 - output)) * -1).sum()
        return r

    def der_cross_entropy(self, output, y):
        return output - y

    def tanh(self, x):
        return np.tanh(x)

    def der_tanh(self, x):
        return 1 / np.square((np.cosh(x)))

    def __init__(self, input_layers=900, hidden_layers=[], output_layers=6,
                 activation_func=sigmoid, der_func=sigmoid_derivation, loss_func=neg_log_likelihood):
        self.input_layers = input_layers
        self.hidden_layer = hidden_layers  # when this is empty array it means single layer
        self.output_layers = output_layers

        self.activation_func = activation_func
        self.der_activation_func = der_func
        self.loss_func = loss_func

        total_layers = [input_layers] + hidden_layers + [output_layers]

        weights = list()  # weights are stored for forward and backprop
        for i in range(len(total_layers) - 1):
            # we create an initial weights.
            # weight shape is (900,6) each feature fully connected to the output
            weight = np.random.rand(total_layers[i], total_layers[i + 1]) / 1000
            weights.append(weight)
        self.weights = weights

        biases = list()
        for i in range(len(total_layers) - 1):
            bias = np.random.rand(total_layers[i + 1]) / 1000
            biases.append(bias)
        self.bias = biases

        activation_outputs = list()  # sigmoid outputs are stored for backprop
        for i in range(len(total_layers)):
            activation_outputs.append(np.zeros(total_layers[i]))
        self.activation_outputs = activation_outputs

        derivation_outputs = list()  # derivations are stored
        for i in range(len(total_layers) - 1):
            derivation_outputs.append(np.zeros((total_layers[i], total_layers[i + 1])))
        self.derivation_outputs = derivation_outputs

        bias_deltas = list()
        for i in range(len(total_layers) - 1):
            bias_deltas.append(np.zeros(total_layers[i + 1]))
        self.bias_deltas = bias_deltas

    def forward(self, inputs):
        # input has 900 feature
        # activation(3)= sig(h2)
        # h2=a(2) x w2 and so on
        activations = inputs
        self.activation_outputs[0] = activations  # first activation is the input
        for i in range(len(self.weights)):
            h1 = np.dot(activations, self.weights[i])
            h1 += self.bias[i]

            activations = self.activation_func(self, h1)
            self.activation_outputs[i + 1] = activations  # first weight calculates the second activation output

        return self.softmax(activations)

    def back_propagation(self, error):
        # yhat-y=error[0,-1,0,0,1,0]
        for i in range(len(self.derivation_outputs) - 1, -1, -1):
            activation_outputs = self.activation_outputs[i + 1]  # these are outputs of layers
            delta = error * self.der_activation_func(self, activation_outputs)  # calculate delta
            transposed_delta = delta.reshape(delta.shape[0], -1).T  # take transpose to calculate derivation
            self.bias_deltas[i] = delta  # bias deriv value

            activation_outputs2 = self.activation_outputs[i]  # this is previous layers output
            activation_outputs2 = activation_outputs2.reshape(activation_outputs2.shape[0], -1)
            self.derivation_outputs[i] = np.dot(activation_outputs2, transposed_delta)  # get the derivation of weight
            error = np.dot(delta, self.weights[i].T)

        return error

    def gradient_descent(self, learning_rate):  # update values
        for i in range(len(self.weights)):
            w = self.weights[i]
            derivation = self.derivation_outputs[i]
            w -= derivation * learning_rate
            self.weights[i] = w

            self.bias[i] -= self.bias_deltas[i] * learning_rate

    def train2(self, train_set, validation_set, epochs=250, learning_rate=0.05, batch_size=4,
               decay_rate=0.99):
        epoch_accuracy = list()
        for i in range(epochs):
            total_error = 0
            e = 0
            j = 0
            error = 0
            for input, target in train_set:
                output = self.forward(input)
                y = np.array([1. if x == target else 0. for x in range(len(output))])
                error += self.der_cross_entropy(output, y)
                loss = self.loss_func(self, output, y)
                total_error += loss
                if (j + 1) % batch_size == 0:
                    self.back_propagation(error / batch_size)
                    self.gradient_descent(learning_rate)
                    error = 0
                if np.argmax(output) == target:
                    e += 1
                j += 1
            learning_rate *= decay_rate  # decay rate
            val_counter = 0
            for val in validation_set:
                o = self.forward(val[0])
                if np.argmax(o) == val[1]:
                    val_counter += 1
            epoch_accuracy.append([i + 1, e / 14034, (val_counter / len(validation_set))
                                      , total_error / len(train_set)])
            if (i + 1) % 50 == 0:
                print("Accuracy is", str(e / 14034))
                print("Error: {} at epoch {}".format(total_error / len(train_set), (i + 1)))
        return epoch_accuracy
