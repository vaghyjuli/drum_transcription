import numpy as np
import matplotlib.pyplot as plt

def nmf(V, R, thresh=0.001, L=1000, W=None, H=None, norm=False, report=False):
    """NMF algorithm with Euclidean distance

    Notebook: C8/C8S3_NMFbasic.ipynb

    Args:
        V (np.ndarray): Nonnegative matrix of size K x N
        R (int): Rank parameter
        thresh (float): Threshold used as stop criterion (Default value = 0.001)
        L (int): Maximal number of iteration (Default value = 1000)
        W (np.ndarray): Nonnegative matrix of size K x R used for initialization (Default value = None)
        H (np.ndarray): Nonnegative matrix of size R x N used for initialization (Default value = None)
        norm (bool): Applies max-normalization of columns of final W (Default value = False)
        report (bool): Reports errors during runtime (Default value = False)

    Returns:
        W (np.ndarray): Nonnegative matrix of size K x R
        H (np.ndarray): Nonnegative matrix of size R x N
        V_approx (np.ndarray): Nonnegative matrix W*H of size K x N
        V_approx_err (float): Error between V and V_approx
        H_W_error (np.ndarray): History of errors of subsequent H and W matrices
    """
    K = V.shape[0]
    N = V.shape[1]
    if W is None:
        W = np.random.rand(K, R)
    if H is None:
        H = np.random.rand(R, N)
    H_W_error = np.zeros((2, L))
    ell = 1
    below_thresh = False
    eps_machine = np.finfo(np.float32).eps
    while not below_thresh and ell <= L:
        H_ell = H
        W_ell = W
        H = H * (W.transpose().dot(V) / (W.transpose().dot(W).dot(H) + eps_machine))
        W = W * (V.dot(H.transpose()) / (W.dot(H).dot(H.transpose()) + eps_machine))
        H_error = np.linalg.norm(H-H_ell, ord=2)
        W_error = np.linalg.norm(W - W_ell, ord=2)
        H_W_error[:, ell-1] = [H_error, W_error]
        if report:
            print('Iteration: ', ell, ', H_error: ', H_error, ', W_error: ', W_error)
        if H_error < thresh and W_error < thresh:
            below_thresh = True
            H_W_error = H_W_error[:, 0:ell]
        ell += 1
    if norm:
        for r in range(R):
            v_max = np.max(W[:, r])
            if v_max > 0:
                W[:, r] = W[:, r] / v_max
                H[r, :] = H[r, :] * v_max
    V_approx = W.dot(H)
    V_approx_err = np.linalg.norm(V-V_approx, ord=2)
    return W, H, V_approx, V_approx_err, H_W_error


R = 3   # instruments
K = 6   # spectral bins
N = 16  # time steps

instrument1 = [2, 0, 3, 0.1, 4, 0.2]
instrument2 = [0, 1, 0.3, 4, 4, 0.1]
instrument3 = [0.1, 3, 0.1, 2, 0, 2]

template_activations = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
activations1 = template_activations.copy()
activations2 = template_activations.copy()
activations3 = template_activations.copy()

activations1[0] = 1
activations1[1] = 0.5
activations1[2] = 0.25
activations1[7] = 1
activations1[8] = 0.5
activations1[9] = 0.25

activations2[2] = 1
activations2[3] = 0.5
activations2[4] = 1
activations2[5] = 0.5

activations3[0] = 1
activations3[1] = 0.5
activations3[4] = 1
activations3[5] = 0.5
activations3[8] = 1
activations3[9] = 0.5
activations3[12] = 1
activations3[13] = 0.5

W_og = np.array([instrument1, instrument2, instrument3], dtype=np.float64)
W_og = W_og.transpose()
H_og = np.array([activations1, activations2, activations3], dtype=np.float64)
V = W_og.dot(H_og)

noise = np.abs(np.random.normal(scale=0.1, size=V.shape))
V_noised = V + noise
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.title('V')
plt.subplot(2, 2, 2)
plt.imshow(V_noised, cmap='hot', interpolation='nearest')
plt.title('V + noise')
plt.show()

W, H, V_approx, V_approx_err, H_W_error = nmf(V=V_noised, R=R, W=W_og)

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(W_og, cmap='hot', interpolation='nearest')
plt.title('W_og')
plt.subplot(2, 2, 2)
plt.imshow(W, cmap='hot', interpolation='nearest')
plt.title('W')
plt.subplot(2, 2, 3)
plt.imshow(H_og, cmap='hot', interpolation='nearest')
plt.title('H_og')
plt.subplot(2, 2, 4)
plt.imshow(H, cmap='hot', interpolation='nearest')
plt.title('H')
plt.show()

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(V, cmap='hot', interpolation='nearest')
plt.title('V_og')
plt.subplot(2, 2, 2)
plt.imshow(V_approx, cmap='hot', interpolation='nearest')
plt.title('V_approx')
plt.show()