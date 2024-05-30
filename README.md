# Perceptron Model

This repository contains a simple implementation of a Perceptron model. The Perceptron is a type of artificial neural network that is used for binary classification tasks. This implementation is designed to be a starting point for those interested in understanding the basics of neural networks and machine learning.

## Overview

The Perceptron model implemented in this repository is a single-layer neural network that can be trained to classify data into two classes. The model uses a linear activation function and is trained using the stochastic gradient descent algorithm.

## Features

- Simple and easy-to-understand implementation of a Perceptron.
- Uses numpy for numerical operations.
- Includes logging for detailed insights into the training process.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/perceptron-model.git
    cd perceptron-model
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install numpy
    ```

## Usage

1. Import the Perceptron class from the `perceptron.py` file:
    ```python
    from perceptron import Perceptron
    ```

2. Initialize the Perceptron model:
    ```python
    eta = 0.01  # Learning rate
    epochs = 100  # Number of training epochs
    perceptron = Perceptron(eta=eta, epochs=epochs)
    ```

3. Prepare your training data:
    ```python
    # Example training data
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([1, 0, 0, 0])  # Labels
    ```

4. Train the model:
    ```python
    perceptron.fit(X, y)
    ```

5. Make predictions:
    ```python
    predictions = perceptron.predict(X)
    print(f"Predictions: {predictions}")
    ```

