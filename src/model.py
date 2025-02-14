import numpy as np
import pdb
import src.random


class Model:
    """
    A wrapper class for a neural network composed of
    layers and a loss function
    """
    def __init__(self, layers, loss, learning_rate=1):
        """
        layers: a list of layers
            each must have a `forward` and `backward` function
        loss: the loss function to use when calling self.backward

        You should not need to edit this function.
        """
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def predict(self, X):
        """
        Helper function to match the scikit-learn API

        You will not need to edit this function.
        """
        return self.forward(X)

    def forward(self, X):
        """
        Take the input and pass it forward through each layer of the network,
        using the `.forward()` function of each layer.

        Return the output of the final layer.
        """

        results = X
        for i in range(len(self.layers)):
            results = self.layers[i].forward(results)
        return results

    def backward(self, pred, y):
        """
        Take the predicted and target outputs and compute the loss.

        Then, beginning with `self.loss` and continuing *backwards*
        through each layer of the network, use the `.backward()`
        function of each layer to perform backpropagation.

        Note: each call to `backward()` in self.layers
            should use self.learning_rate

        Returns None
        """
        loss = self.loss.forward(pred, y)
        grad = self.loss.backward()
        grad = np.dot(np.identity(grad.shape[0]), grad)
        

        #K Lo K
        for i in range(len(self.layers)-1,-1,-1):
            grad = self.layers[i].backward(grad, lr = self.learning_rate)
            """
            if i % 2 == 0:
                
                pdb.set_trace()
                assert self.layers[i].weights.shape == grad.shape
                self.layers[i].weights =- grad *self.learning_rate
                pdb.set_trace()
                """


        

    def fit(self, X, y, max_iter=10000):
        """
        Train the model on the data for `max_iter` iterations.
        For each iteration, call `self.forward` and then `self.backward`
            to make a prediction and then update each layer's weights.

        This function should always run for `max_iter` iterations;
            don't stop even if the gradients are negligibly small.

        Returns None
        """
           #K Lo K
        for i in range(max_iter):
            self.forward(X)
            self.backward(self.predict(X),y)


def main():
    '''
    A simple MLP to fit the xor dataset.
    This should run and get 100% accuracy after you finish
        implementing the functions in this file.
    '''
    from src.layers import FullyConnected, SigmoidActivation
    from src.loss import BinaryCrossEntropyLoss
    src.random.rng.seed()

    # xor dataset
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    layers = [
      FullyConnected(2, 8), SigmoidActivation(),
      FullyConnected(8, 1), SigmoidActivation(),
    ]

    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=0.1)
    model.fit(X, y, max_iter=10000)
    preds = model.forward(X)
    print("{:.0f}% accuracy".format(100 * np.mean((preds > 0.5) == y)))


if __name__ == "__main__":
    main()
