### YOUR CODE HERE

import numpy as np 
import matplotlib.pyplot as plt
from neural_network import load_data_large, train_and_valid


def partB():
    x = [i for i in range(100)]
    (X_train, y_train, X_val, y_val) = load_data_large()
    num_epochs = 100
    hidden_units = 50
    init_rand = True
    lr = 0.001
    losses_train, losses_val, _, _, _, _ = train_and_valid(X_train, y_train, X_val, y_val, num_epochs, hidden_units, init_rand, lr)
    plt.plot(x, losses_train, color="blue")
    plt.plot(x, losses_val, color="green")
    plt.show()


def partA(): 
    num_epochs = 100 
    hidden = [5,20,50,100,200]
    (X_train, y_train, X_val, y_val) = load_data_large() 
    init_rand = True
    learning_rate = 0.01

    y_axis_train = [] 
    y_axis_valid = [] 
    for num_hidden in hidden: 
        (losses_train, losses_val,
        train_error, valid_error,
        y_hat_train, y_hat_valid ) = train_and_valid(X_train, y_train, X_val, y_val, num_epochs, num_hidden, init_rand, learning_rate)
        y_axis_train += [losses_train[-1]]
        y_axis_valid += [losses_val[-1]]

    plt.plot(hidden, y_axis_train, color = "blue") 
    plt.plot(hidden, y_axis_valid, color = "green")
    plt.show()



partB()