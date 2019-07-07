import numpy

class NeuralNetwork():

    def __init__(self):

        self.x = 5      # input layer
        self.y = 3      # output layer
        self.h = 7      # hidden layer

        self.neurons = {}
        self.neurons['W1'] = numpy.random.randn(self.h, self.x) * 0.1
        self.neurons['b1'] = numpy.random.randn(self.h, 1) * 0.1

        self.neurons['W2'] = numpy.random.randn(self.y, self.h) * 0.1
        self.neurons['b2'] = numpy.random.randn(self.y, 1) * 0.1

    def select_action(self, observation):

        def relu(z):
            # Activation function
            return z * (z > 0)

        def feed_forward(observation):
            # Feed observations into NN
            z1 = numpy.dot(self.neurons['W1'], observation) + self.neurons['b1']
            a1 = relu(z1)
            z2 = numpy.dot(self.neurons['W2'], a1) + self.neurons['b2']
            a2 = relu(z2)

            return numpy.argmax(a2)

        observation = observation.reshape(self.x, 1)
        action = feed_forward(observation)

        return action

