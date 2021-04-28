# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        self.input_dim = input_dim
        self.load_data()
        self.hidden_layer = hidden_layer
        self.aj = [[], []]
        if hidden_layer:
            self.weights = [np.random.rand(self.input_dim + 1, self.hidden_units) - 0.5, np.random.rand(self.hidden_units + 1) - 0.5]
            self.delta = [[], [], []]
        else:
            self.weights = np.random.rand(self.input_dim + 1) - 0.5
            self.delta = []

        self.a = np.zeros(input_dim)
        self.delta_j = np.zeros(input_dim)


    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def der_sigmoid(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))
    
    def train(self) -> None:

        """Run the backpropagation algorithm to train this neural network"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class
        while self.epochs:
            for x, y in zip(self.x_train, self.y_train):
                self.a = np.append(x, 1)
                if self.hidden_layer:
                    inj_x = self.a @ self.weights[0]
                    self.aj[0] = self.sigmoid(inj_x) 
                    self.aj[0] = np.append(self.aj[0], 1)
                    inj_out = self.aj[0] @ self.weights[1]
                    self.aj[1] = self.sigmoid(inj_out)
                    self.delta[0] = self.der_sigmoid(inj_out) * (y - self.aj[1])
                    inj_x = np.append(inj_x, 1)
                    self.delta[1] = self.der_sigmoid(inj_x) * self.weights[1]*self.delta[0]
                    self.weights[1][:self.hidden_units] = self.weights[1][:self.hidden_units] + self.lr * self.aj[0][:25] * self.delta[0]
                    self.weights[0][:self.input_dim] = self.weights[0][:self.input_dim] + self.lr * np.matmul(self.a[:30].reshape(30, 1), self.delta[1][:25].reshape(1, 25))
                    self.weights[1][self.hidden_units] = self.weights[1][25] + self.lr*self.delta[0]
                    self.weights[0][self.input_dim] = self.weights[0][self.input_dim] + self.lr*self.delta[1][:25]
                 
                else: 
                    inj_x = self.a @ self.weights
                    self.aj[0] = self.sigmoid(inj_x)
                    self.delta = self.der_sigmoid(inj_x) * (y - self.aj[0])
                    self.weights = self.weights + self.lr*self.a*self.delta

            self.epochs -= 1
      
        pass

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # TODO: Implement the forward pass.
        if self.hidden_layer:
            x = np.append(x, 1)
            inj_x = np.dot(x, self.weights[0])
            aj = [self.sigmoid(i) for i in inj_x]
            aj = np.append(aj, 1)
            inj_out = np.dot(aj, self.weights[1])
            return self.sigmoid(inj_out)
        else:
            x = np.append(x, 1)
            inj_x = x @ self.weights
            return self.sigmoid(inj_x)

class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
    # nn = NeuralNetwork(30, 1)
    # nn.train()
