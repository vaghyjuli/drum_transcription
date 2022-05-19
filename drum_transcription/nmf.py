import numpy as np

EPS = 2.0 ** -52

def NMF(V, W, L = 1000, threshold = 0.001, fixW=True):
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
    
    K, R = W.shape
    K, N = V.shape
    H = np.random.rand(R, N)

    #EPS = np.finfo(np.float32).eps

    W_diff = 0
    for iteration in range(L):
        H_prev = H
        H = H * (W.transpose().dot(V) / (W.transpose().dot(W).dot(H) + EPS))
        if not fixW:
            W_prev = W
            W = W * (V.dot(H.transpose()) / (W.dot(H).dot(H.transpose()) + EPS))
            W_diff = np.linalg.norm(W - W_prev, ord=2)
        H_diff = np.linalg.norm(H - H_prev, ord=2)
        if H_diff < threshold and W_diff < threshold:
            break

    V_approx = W.dot(H)
    V_approx_err = np.linalg.norm(V - V_approx, ord=2)
    #print(f"NMF finished with V_approx_err={V_approx_err}\n")
    return V_approx, W, H