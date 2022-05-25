import numpy as np
from copy import deepcopy

EPS = 2.0 ** -52

def NMF(V, W_init, L = 1000, threshold = 0.001, fixW=True, addedCompW=2, initH="random", beta=4):
    """
        Non-Negative Matrix Factor Deconvolution with Kullback-Leibler-Divergence.

        Parameters:
            V (np.ndarray) : A 2D numpy array of size K x N, representing the magnitude
                spectrogram with K spectral bands on N time steps.
            W (np.ndarray) : A 3D numpy array of size K x R x T, representing template
                magnitude spectrograms with K spectral bands on T time steps, for each
                of the R instruments.
            L (int) : The number of NMFD iterations.
            threshold (float) : If the difference between W and W', or between H and H'
                is < threshold, the gradient descent stops, 
            fixW (bool) : Indicator for whether the template matrix should be fully
                adaptive (= False) or fixed (= True).

        Returns:
            V_approx (np.ndarray) : A 2D numpy array of size K x N,  representing the
                magnitude spectrogram approximated by the NMFD components.
            W (np.ndarray) : A 3D numpy array of size K x R x T, representing the
                (adapted) template magnitude spectrograms.
            H (np.ndarray) : A 2D numpy array of size R x N, representing the activations
                for each of the R instruments over N time steps.
    """
    K, N = V.shape
    K, R = W_init.shape

    # add randomoly initialized noise components to W
    W_init = np.append(W_init, np.ones((K, addedCompW)) + EPS, axis=1)
    R += addedCompW
    
    if initH == "random":
        H = np.random.rand(R, N)
    elif initH == "uniform":
        H = np.ones(R, N)

    W = deepcopy(W_init)
    onesMatrix = np.ones((K, N))

    for iteration in range(L):
        V_approx = W.dot(H)
        Q = V / (V_approx + EPS)

        H_prev = deepcopy(H)
        H = H * (W.transpose().dot(Q) / (W.transpose().dot(onesMatrix) + EPS))
        W_prev = deepcopy(W)
        W = W * (Q.dot(H.transpose()) / (onesMatrix.dot(H.transpose()) + EPS))

        if fixW == "fixed":
            W[:, :R-addedCompW] = W_init[:, :R-addedCompW]
        elif fixW == "semi":
            alpha = (1 - iteration / L)**beta
            W[:, :R-addedCompW] = alpha * W_init[:, :R-addedCompW] + (1 - alpha) * W[:, :R-addedCompW]

        W_diff = np.linalg.norm(W - W_prev, ord=2)
        H_diff = np.linalg.norm(H - H_prev, ord=2)
        if H_diff < threshold and W_diff < threshold:
            break

    V_approx = W.dot(H)
    return V_approx, W, H