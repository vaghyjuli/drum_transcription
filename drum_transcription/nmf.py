import numpy as np

def NMF(V, W_init, fixW=True, norm=False, thresh = 0.001, L = 1000, report=False):
    K, R = W_init.shape
    K, N = V.shape
    W = W_init
    H = np.random.rand(R, N)
    H_W_error = np.zeros((2, L))
    ell = 1
    below_thresh = False
    eps_machine = np.finfo(np.float32).eps
    while not below_thresh and ell <= L:
        H_ell = H
        W_ell = W
        H = H * (W.transpose().dot(V) / (W.transpose().dot(W).dot(H) + eps_machine))
        if not fixW:
            W = W * (V.dot(H.transpose()) / (W.dot(H).dot(H.transpose()) + eps_machine))
        H_error = np.linalg.norm(H - H_ell, ord=2)
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
    V_approx_err = np.linalg.norm(V - V_approx, ord=2)
    #print(f"NMF finished with V_approx_err={V_approx_err}\n")
    return V_approx, W, H