import numpy as np
from copy import deepcopy

EPS = 2.0 ** -52

def NMFD(V, W_init, params, L=50):
    """
        Non-Negative Matrix Factor Deconvolution with Kullback-Leibler-Divergence.

        Parameters:
            V (np.ndarray) : A 2D numpy array of size K x N, representing the magnitude
                spectrogram with K spectral bands on N time steps.
            W (np.ndarray) : A 3D numpy array of size K x R x T, representing template
                magnitude spectrograms with K spectral bands on T time steps, for each
                of the R instruments.
            L (int) : The number of NMFD iterations.
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

    # num of spectral bands, num of NMFD components, num of time frames in the component templates
    K, R, T = W_init.shape
    # num of spectral bands, num of time frames in the full spectrogram
    K, N = V.shape

    W_init = np.append(W_init, np.random.rand(K, params["addedCompW"], T) + EPS, axis=1)
    R += params["addedCompW"]

    # initalize the activation matrix
    if params["initH"] == None or params["initH"] == "uniform":
        H = np.ones((R, N))
    elif params["initH"] == "random":
        H = np.random.rand(R, N)

    if params["beta"] == None:
        print("beta 4 nmfd")
        params["beta"] = 4
    
    W = deepcopy(W_init)

    # helper matrix of all ones (denoted as J in eq (5,6) in [2])
    onesMatrix = np.ones((K, N))

    for iteration in range(L):
        H_prev = H
        V_approx = convModel(W, H)

        # compute the ratio of the input to the model
        Q = V / (V_approx + EPS)

        # accumulate activation updates here
        multH = np.zeros((R, N))

        # go through all template frames
        for t in range(T):
            # use tau for shifting and t for indexing
            tau = deepcopy(t)

            # The update rule for W as given in eq. (5) in [2]
            # pre-compute intermediate, shifted and transposed activation matrix
            transpH = shiftOperator(H, tau).T

            # multiplicative update for W
            multW = Q @ transpH / (onesMatrix @ transpH + EPS)
            W[:, :, t] *= multW

            if params["fixW"] == "fixed":
                W[:, :R-params["addedCompW"], t] = W_init[:, :R-params["addedCompW"], t]            
            elif params["fixW"] == "semi":
                alpha = (1 - iteration / L)**params["beta"]
                W[:, :R-params["addedCompW"], t] = alpha * W_init[:, :R-params["addedCompW"], t] + (1 - alpha) * W[:, :R-params["addedCompW"], t]

            # The update rule for W as given in eq. (6) in [2]
            # pre-compute intermediate matrix for basis functions W
            transpW = W[:, :, t].T

            # compute update term for this tau
            addW = (transpW @ shiftOperator(Q, -tau)) / (transpW @ onesMatrix + EPS)

            # accumulate update term
            multH += addW

        # multiplicative update for H, with the average over all T template frames
        H *= multH

    V_approx = convModel(W, H)
    return V_approx, W, H


def convModel(W, H):
    """
        Convolutive NMF model implementing the eq. (4) from [2]. Note that it can
        also be used to compute the standard NMF model in case the number of time
        frames of the templates equals one.
        Parameters
        ----------
        W: array-like
            Tensor holding the spectral templates which can be interpreted as a set of
            spectrogram snippets with dimensions: numBins x numComp x numTemplateFrames
        H: array-like
            Corresponding activations with dimensions: numComponents x numTargetFrames
        Returns
        -------
        V_approx: array-like
            Approximated spectrogram matrix
    """
    # the more explicit matrix multiplication will be used
    K, R, T = W.shape
    R, N = H.shape

    # initialize with zeros
    V_approx = np.zeros((K, N))

    # this is doing the math as described in [2], eq (4)
    # the alternative conv2() method does not show speed advantages

    for k in range(T):
        multResult = W[:, :, k] @ shiftOperator(H, k)
        V_approx += multResult

    V_approx += EPS

    return V_approx


def shiftOperator(A, shiftAmount):
    """
        Shift operator as described in eq. (5) from [2]. It shifts the columns
        of a matrix to the left or the right and fills undefined elements with
        zeros.
        Parameters
        ----------
        A: array-like
            Arbitrary matrix to undergo the shifting operation
        shiftAmount: int
            Positive numbers shift to the right, negative numbers
            shift to the left, zero leaves the matrix unchanged
        Returns
        -------
        shifted: array-like
            Result of this operation
    """
    # Get dimensions
    numRows, numCols = A.shape

    # Limit shift range
    shiftAmount = np.sign(shiftAmount) * min(abs(shiftAmount), numCols)

    # Apply circular shift along the column dimension
    shifted = np.roll(A, shiftAmount, axis=-1)

    if shiftAmount < 0:
        shifted[:, numCols + shiftAmount: numCols] = 0

    elif shiftAmount > 0:
        shifted[:, 0: shiftAmount] = 0

    return shifted