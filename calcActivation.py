# function to calculate the activations
def calcActivation(model, X, act):
    '''
    input:  model, NN model
            X, input data, (m,n)
            act, activation used for the layers
    output: activations for the first and second layer
            a1(m,n1), a2(m,n2)
    '''
    
    # calculation weights for the layer
    h1 = model.layers[1]
    h2 = model.layers[2]
    
    w1 = h1.get_weights()[0]
    b1 = h1.get_weights()[1]
    w2 = h2.get_weights()[0]
    b2 = h2.get_weights()[1]
    
    # calculate activation for h1 layer
    # a = g(z) = g(wx + b)
    # (nl,m)   (nl,n)(n,m)  (nl,1)

    # reshape to right dimensions
    b1 = b1.reshape((-1,1))
    b2 = b2.reshape((-1,1))

    # calculate z
    z1 = w1.T @ X.T + b1
    
    if act == 'relu':
        # relu function for layer activations
        a1 = np.maximum(0, z1)
        # activation for h2 layer
        z2 = w2.T @ a1 + b2
        a2 = np.maximum(0, z2)
    elif act == 'tanh':
        a1 = np.tanh(z1)
        z2 = w2.T @ a1 + b2
        # (n2,m) = (n2,n1) @ (n1,m) + (n2,1)
        a2 = np.tanh(z2)
    
    return {'a1' : a1.T, 'a2' : a2.T}
