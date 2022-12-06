import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

# Useful values
    m = np.shape(X)[0]              #number of examples
    
# You need to return the following variables correctly 
    p = np.zeros(m);

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#

    # Implements the feedforward computation that computes htheta(x(i)) for every example x(i) in X
    # You need to add a column of 1s to the matrix --> the first column is the bias term
    ones = np.ones([m, 1])
    X = np.column_stack((ones, X))

    # Matrices Theta1 and Theta2 contain the parameters for each unit in rows
    # The first row of theta1 corresponds to the fist hidden unit in the second layer
    # When you compute Z(2) = theta(1)a(1) be sure that you index X correctly so that you get a(i) as a column vector
    # REMEMBER: Theta 1 and Theta2 are the trained weights
    # Loop through all instances
    for i in range(m):
        a1 = X[i]
        z2 = np.dot(Theta1, a1)
        a2 = sigmoid(z2)
        a2 = np.concatenate((np.array([1, ]), a2), axis=None)
        z3 = np.dot(Theta2, a2)
        a3 = sigmoid(z3)

        p[i] = np.round(np.argmax(a3) + 1)


    # Returns the neural network predictions
    return p

# =========================================================================

