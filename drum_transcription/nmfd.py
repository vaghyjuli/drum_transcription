import numpy as np
from copy import deepcopy

EPS = 2.0 ** -52

def NMFD(V, W_init, L=50, fixW=False):
    """
        Non-Negative Matrix Factor Deconvolution with Kullback-Leibler-Divergence
        and fixable components.
        Parameters
        ----------
        V: array-like
            Matrix that shall be decomposed (typically a magnitude spectrogram of dimension
            numBins x numFrames)
        L: int
            Number of NMFD iterations
        T: int
            Number of time frames for 2D-templates
        R: int
            Number of NMF components        
        Returns
        -------
        W: array-like
            List with the learned templates
        H: array-like
            Matrix with the learned activations
        nmfdV: array-like
            List with approximated component spectrograms
        costFunc: array-like
            The approximation quality per iteration
        tensorW: array-like
            If desired, we can also return the tensor
    """
    # use parameter nomenclature as in [2]
    K, R, T = W_init.shape
    K, N = V.shape
    initH = np.random.rand(R, N)
    tensorW = np.zeros((K, R, T))
    costFunc = np.zeros(L)

    # stack the templates into a tensor
    for r in range(R):
        tensorW[:, r, :] = W_init[:, r, :]

    # the activations are matrix shaped
    H = deepcopy(initH)

    # create helper matrix of all ones (denoted as J in eq (5,6) in [2])
    onesMatrix = np.ones((K, N))

    for iteration in range(L):
        # compute first approximation
        V_approx = convModel(tensorW, H)

        # store the divergence with respect to the target spectrogram
        costMat = V * np.log(1.0 + V/(V_approx+EPS)) - V + V_approx
        costFunc[iteration] = costMat.mean()

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

            if not fixW:
                # multiplicative update for W
                multW = Q @ transpH / (onesMatrix @ transpH + EPS)
                tensorW[:, :, t] *= multW

            # The update rule for W as given in eq. (6) in [2]
            # pre-compute intermediate matrix for basis functions W
            transpW = tensorW[:, :, t].T

            # compute update term for this tau
            addW = (transpW @ shiftOperator(Q, -tau)) / (transpW @ onesMatrix + EPS)

            # accumulate update term
            multH += addW

        # multiplicative update for H, with the average over all T template frames
        H *= multH / T

        # normalize templates to unit sum
        #normVec = tensorW.sum(axis=2).sum(axis=0)

        #tensorW *= 1.0 / (EPS+np.expand_dims(normVec, axis=1))

    W = list()
    nmfdV = list()

    # compute final output approximation
    for r in range(R):
        W.append(tensorW[:, r, :])
        nmfdV.append(convModel(np.expand_dims(tensorW[:, r, :], axis=1), np.expand_dims(H[r, :], axis=0)))

    return W, H, nmfdV, costFunc, tensorW


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
        lamb: array-like
            Approximated spectrogram matrix
    """
    # the more explicit matrix multiplication will be used
    K, R, T = W.shape
    R, N = H.shape

    # initialize with zeros
    lamb = np.zeros((K, N))

    # this is doing the math as described in [2], eq (4)
    # the alternative conv2() method does not show speed advantages

    for k in range(T):
        multResult = W[:, :, k] @ shiftOperator(H, k)
        lamb += multResult

    lamb += EPS

    return lamb


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

    else:
        pass

    return shifted