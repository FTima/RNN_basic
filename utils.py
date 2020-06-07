import numpy as np 
from rnn_utils import * 

def rnn_cell_forward(xt, a_prev, parameters):

    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,xt)+ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(Wya,a_next)+by)
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    
    # initialize "a" and "y_pred" with zeros (≈2 lines)
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    
    # Initialize a_next (≈1 line)
    a_next = a0
    
    # loop over all time-steps of the input 'x' (1 line)
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache (≈2 lines)
        xt = x[:,:,t]
        a_next, yt_pred, cache = rnn_cell_forward(xt,a_next,parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)
        
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    
    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈1 line)
    concat = np.concatenate(a_prev,xt)

    # Compute values for ft (forget gate), it (update gate),
    # cct (candidate value), c_next (cell state), 
    # ot (output gate), a_next (hidden state) (≈6 lines)
    ft = sigmoid(np.dot(Wf,concat)+bf)        # forget gate
    it = sigmoid(np.dot(Wi,concat)+bi)        # update gate
    cct = np.tanh(np.dot(Wc,concat)+bc)       # candidate value
    c_next = ft*c_prev+it*cct    # cell state
    ot = sigmoid(np.dot(Wo,concat)+bo)        # output gate
    a_next = ot*np.tanh(c_next)    # hidden state
    
    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(Wy,a_next)+by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    
    # Initialize "caches", which will track the list of all the caches
    caches = []
    
    Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = np.shape(x)
    n_y, n_a = np.shape(Wy)
    
    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros((n_a,m,T_x))
    c = np.zeros((n_a,m,T_x))
    y = np.zeros((n_y,m,T_x))
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = c[:,:,0]
    
    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:,:,t]
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(xt,a_next,c_next,parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the next cell state (≈1 line)
        c[:,:,t]  = c_next
        # Save the value of the prediction in y (≈1 line)
        y[:,:,t] = yt
        # Append the cache into caches (≈1 line)
        caches.append(cache)
        
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches