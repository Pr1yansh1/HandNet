import numpy as np 
import math



def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    return np.matmul(p,input)

def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    func = np.vectorize(lambda x : 1/(1 + math.exp(-x)))
    return func(a)

def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    exp = np.vectorize(lambda x: math.exp(x))
    exp_v = exp(b)
    func = np.vectorize(lambda x : x/sum(exp_v))
    res = func(exp_v)
    return res


def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    elem = y_hat[hot_y,0]
    return -math.log(elem)


def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: beta WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """

    a = linearForward(x,alpha) 
    z = sigmoidForward(a) 
    #need to append value of 1 for the first entry of z
    z = np.vstack([[1.0], z])
    b = linearForward(z,beta)
    y_hat = softmaxForward(b) 
    J = crossEntropyForward(y,y_hat)
    return (x,a,z,b,y_hat,J)


def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    y_hat[hot_y, 0] -= 1
    return y_hat


def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions.
    """
    ok = np.transpose(prev)
    grad_param = np.matmul(grad_curr,ok ) 
    #remove the bias term i.e. the first column 
    beta_star = p[:,1:]
    v = np.transpose(beta_star)
    grad_prev1 = np.matmul(v, grad_curr)
    return grad_param, grad_prev1


def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    def helper(x,y): 
        return x*(y**2)*((1/y) - 1)

    func = np.vectorize(helper)

    return func(grad_curr, curr)


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    TIP: Make sure you're accounting for the changes due to the bias term
    """

    #returns partial derivative vector wrt b
    
    g_b = softmaxBackward(y, y_hat) 
    g_beta, g_z = linearBackward(z, beta, g_b) 
    g_a = sigmoidBackward(np.delete(z,0,0),g_z) 
    g_alpha, g_x = linearBackward(x,alpha,g_a)
    return (g_alpha, g_beta, g_b,g_z,g_a)



def SGD(X_train, y_train, X_val, y_val, hidden_units, num_epochs, init_rand, learning_rate):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param hidden_units: Number of hidden units
    :param num_epochs: Number of epochs
    :param init_rand:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    K = 10 
    M = len(X_train[0]) 
    
    def J_SGD(alpha, beta, x,y): 
        x = np.vstack([[1], x.reshape((M, -1))])
        (_,_,_,_, _, J) = NNForward(x, y, alpha, beta)
        return J

    D = hidden_units
    alpha = np.zeros((D,M))
    beta = np.zeros((K,D))
    alpha = np.append(np.zeros((D, 1)), alpha, axis=1)
    beta = np.append(np.zeros((K, 1)), beta, axis=1)
    if init_rand: 
        func_init = np.vectorize(lambda x: np.random.uniform(-0.1,0.1)) 
        alpha = func_init(alpha) 
        beta = func_init(beta)
    losses_train = [] 
    losses_val = [] 
    N_train = len(X_train)
    N_val = len(X_val)
    func = np.vectorize(lambda x: x*math.log(x))
    for i in range(num_epochs): 
        # print("hi")
        # print(N_train)
        for j in range(N_train): 
            # print(j)
            x = X_train[j] 
            y = y_train[j]
            x = np.vstack([[1], x.reshape((M, -1))])
            (x,_, z,_, y_hat, J) = NNForward(x,y,alpha,beta) 
            # print(np.shape(x), np.shape(y), np.shape(alpha), np.shape(beta), np.shape(z),np.shape(y_hat))
            (g_alpha, g_beta, g_b,g_z,g_a) = NNBackward(x,y,alpha,beta,z,y_hat)
            # print("hi2")
            alpha = np.subtract(alpha,learning_rate*g_alpha)
            beta = np.subtract(beta,learning_rate*g_beta)
            # print("hi1")
        
        J_train = [] 
        J_val = [] 
        for p in range(N_train): 
            J_train += [J_SGD(alpha,beta, X_train[p], y_train[p])]
        for p in range(N_val):
            J_val += [J_SGD(alpha,beta, X_val[p], y_val[p])]
        losses_train += [sum(J_train)/ len(J_train)]
        losses_val += [sum(J_val)/len(J_val)]
        
    return alpha,beta,losses_train,losses_val 



            

def prediction(X_train, y_train, X_val, y_val, tr_alpha, tr_beta):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data (list)
        - y_hat_valid: predicted labels for validation data (list)
    """
    pass

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epochs, num_hidden, init_rand, learning_rate):
    """ 
    Main function to train and validate your neural network implementation.

    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param num_epochs: Number of epochs to train (i.e. number of loops through the training data).
    :param num_hidden: Number of hidden units.
    :param init_rand: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Float value specifying the learning rate for SGD.

    :return: a tuple of the following six objects, in order:
        - loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        - loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        - err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        - err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        - y_hat_train: A list of integers representing the predicted labels for training data
        - y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE
    loss_per_epoch_train = []
    loss_per_epoch_val = []
    err_train = None
    err_val = None
    y_hat_train = None
    y_hat_val = None
 
    return (loss_per_epoch_train, loss_per_epoch_val,
            err_train, err_val, y_hat_train, y_hat_val)
